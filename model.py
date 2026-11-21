import torch
import torch.nn as nn
import torch.nn.functional as F

# Model must have a create layer function - from scratch but they take the average of random snapshot weights
def average_weights(state_dicts):
    avg_state_dict = {}
    for key in state_dicts[0]:
        stacked_params = torch.stack([sd[key] for sd in state_dicts],dim=0)

        if stacked_params.is_floating_point():
            avg_state_dict[key] = stacked_params.mean(dim=0)
        else:
            avg_state_dict[key] = state_dicts[0][key]

    return avg_state_dict

class RTHead(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=False),  # 32x32
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            # Depthwise separable refinement block
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, groups=96, bias=False),
            nn.Conv2d(96, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        return F.silu(self.proj(pooled))


class CHead(nn.Module):
    def __init__(self, out_dim: int = 384):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
        )
        self.stage1 = self._make_stage(64, 64, num_blocks=2, stride=1)
        self.stage2 = self._make_stage(64, 128, num_blocks=3, stride=2)
        self.stage3 = self._make_stage(128, 256, num_blocks=6, stride=2)
        self.stage4 = self._make_stage(256, 512, num_blocks=2, stride=2)
        self.proj = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Linear(512, out_dim),
        )

    @staticmethod
    def _make_stage(in_c: int, out_c: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = []
        for i in range(num_blocks):
            s = stride if i == 0 else 1
            c_in = in_c if i == 0 else out_c
            layers.append(
                nn.Sequential(
                    nn.Conv2d(c_in, out_c, kernel_size=3, stride=s, padding=1, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(out_c),
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for stage in (self.stage1, self.stage2, self.stage3, self.stage4):
            for block in stage:
                residual = block[0](h) if block[0].stride != (1, 1) or block[0].in_channels != block[0].out_channels else h
                h = F.silu(residual + block(h))
        pooled = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return self.proj(pooled)


class MultiHead(nn.Module):
    def __init__(self,n_c=4,n_rt=16,out_dim=200):
        super().__init__()
        self.c_heads = nn.ModuleDict()
        self.rt_heads = nn.ModuleDict()
        
        for x in range(n_c):
            self.c_heads[str(x)]=CHead(out_dim=out_dim)

        for x in range(n_rt):
            self.rt_heads[str(x)]=RTHead(out_dim=out_dim)

    def add_rt(self,snapshots):
        snapshot_mean = average_weights(snapshots) # In either 2:1 or 3:1 ratio (RT to C)
        rt_layer = RTHead()
        rt_layer.load_state_dict(snapshot_mean)
        new_idx = int(next(reversed(self.rt_heads)))+1
        self.rt_heads[str(new_idx)] = rt_layer

    def remove_rt(self,idx):
        del self.rt_heads[idx]

    @torch.no_grad()
    def eval_head(self,x,targets,criterion_c,criterion_rt):
        perf_metrics = {
            "c_heads":{},
            'rt_heads':{}
        }

        for idx,head in self.c_heads.items():
            outputs = head(x)
            loss = criterion_c(outputs,targets).item()
            perf_metrics['c_heads'][idx] = loss
        
        for idx,head in self.rt_heads.items():
            outputs = head(x)
            loss = criterion_rt(outputs,targets).item()
            perf_metrics['rt_heads'][idx] = loss
        
        return perf_metrics
    
    @torch.no_grad()
    def get_c_avg(self):
        states = [head.state_dict() for head in model.c_heads.values()]
        avg_weights = average_weights(states)
        return avg_weights

    def forward(self,x):
        c_out = [head_c(x) for head_c in self.c_heads.values()]
        rt_out = [head_rt(x) for head_rt in self.rt_heads.values()]
        return c_out,rt_out

