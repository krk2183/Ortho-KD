from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import transforms
from tqdm.auto import tqdm
from model import MultiHead
from tinyimagenet import TinyImageNet
from governor import Governor

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gvnr = Governor()

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizeCrop(64,scale=(0.5,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_dir = Path("./data/tinyimagenet")

train_dataset = TinyImageNet(data_dir, split="train", transform=train_transform)
val_dataset = TinyImageNet(data_dir, split="val", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False)

""""
IMPLEMENT: state_dict saving (snapshots) in 3:1/2:1 ratio
"""
model = MultiHead(n_c=4, n_rt=16, out_dim=200)
model = model.to(device, non_blocking=True)

optimizer = torch.optim.AdamW([
    {
        "params": model.c_heads.parameters(),
        "lr": 5e-4,
        "weight_decay": 5e-4,
    },
    {
        "params": model.rt_heads.parameters(),
        "lr": 3e-3,
        "weight_decay": 5e-3,
    },
])
criterion_c, criterion_rt = (
    nn.CrossEntropyLoss(label_smoothing=0.05),
    nn.CrossEntropyLoss(),
)

scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=EPOCHS, T_mult=1, eta_min=1e-6)
patience_limit = 1 # recommended 20% of num epochs
elemination_patience = 2
EPOCHS = 80
snapshots = []
best_c_loss = float('inf')
rt_temp_loss = float('inf')
head_patience = {
    'rt_heads': {name: {"best_loss": float("inf"), "patience": 0} for name in model.rt_heads.keys()}
}
for epoch in range(EPOCHS):
    training_loss_c, training_loss_rt = 0.0, 0.0
    model.train()
    train_pbar = tqdm(train_loader,desc=f"Epoch [{epoch + 1:02d}/{EPOCHS:02d}] Train",leave=False,)

    for step, (image, label) in enumerate(train_pbar, start=1):
        image = image.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits_c,features_c, rt_logits,rt_features = model(image)

        loss_c_base = sum(criterion_c(head, label) for head in logits_c) / len(logits_c)
        loss_c = loss_c_base + (0.05 * gvnr.c_diversity(features_c))
        loss_rt = gvnr.compute_rt_loss(rt_logits,rt_features,label,criterion_rt)

        loss_c.backward()
        loss_rt.backward()
        optimizer.step()

        training_loss_c += loss_c.item()
        training_loss_rt += loss_rt.item()

        train_pbar.set_postfix({
            "loss_c": f"{training_loss_c / step:.4f}",
            "loss_rt": f"{training_loss_rt / step:.4f}",
        })

    avg_train_c = training_loss_c / len(train_loader)
    avg_train_rt = training_loss_rt / len(train_loader)

    model.eval()
    val_loss_c_sum, val_loss_rt_sum = 0.0, 0.0
    correct, total = 0, 0

    val_pbar = tqdm(
        val_loader,
        desc=f"Epoch [{epoch + 1:02d}/{EPOCHS:02d}] Val  ",
        leave=False,
    )
    # Validation loop
    with torch.no_grad():
        for step, (image, label) in enumerate(val_pbar, start=1):
            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            logits_c,features_c, rt_logits,rt_features = model(image)

            loss_c = sum(criterion_c(head, label) for head in logits_c) / len(logits_c)
            loss_rt = gvnr.compute_rt_loss(rt_features,rt_logits,label,criterion_rt)

            val_loss_c_sum += loss_c.item()
            val_loss_rt_sum += loss_rt.item()

            stacked_c_logits = torch.stack(logits_c,dim=0)
            mean_logits_c = stacked_c_logits.mean(dim=0)

            preds = mean_logits_c.argmax(dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)

            val_pbar.set_postfix({
                "val_c": f"{val_loss_c_sum / step:.4f}",
                "val_rt": f"{val_loss_rt_sum / step:.4f}",
                "acc": f"{100.0 * correct / max(total, 1):.2f}%",
            })

    avg_val_c = val_loss_c_sum / len(val_loader)
    avg_val_rt = val_loss_rt_sum / len(val_loader)
    val_acc = 100.0 * correct / max(total, 1)

    tqdm.write(
        f"Epoch [{epoch + 1:02d}/{EPOCHS:02d}] | "
        f"Train Loss (c/rt): {avg_train_c:.4f} / {avg_train_rt:.4f} | "
        f"Val Loss (c/rt): {avg_val_c:.4f} / {avg_val_rt:.4f} | "
        f"Val Acc: {val_acc:.2f}%"
    )

    # layer pruning & addition, alongside per-layer performance tracking
    sample_x, sample_target = val_loader.dataset[torch.randint(0, len(val_loader.dataset), (1,)).item()]
    sample_x = sample_x.unsqueeze(0).to(device)
    sample_target = torch.tensor([sample_target], device=device)
    epoch_metrics = model.eval_head(sample_x,sample_target,criterion_c,criterion_rt)
    model, head_patience = gvnr.clean_model(model,epoch_metrics,head_patience,snapshots,patience_limit=patience_limit)

    rt_losses = epoch_metrics['rt_heads']
    sorted_rt = sorted(rt_losses.items(),key=lambda item: item[1])

    # snapshotting
    if (epoch+1)>=int(EPOCHS*0.2) and len(sorted_rt) >= 2:
        c_vals = epoch_metrics["c_heads"].values()
        mean_c_loss = sum(c_vals) / len(c_vals) 
        if mean_c_loss<best_c_loss:
            best_c_loss=mean_c_loss
            avg_weights = model.get_c_avg()
            snapshots.append(avg_weights)

            # get the best 2 RT heads
            best_idx, best_loss = sorted_rt[0]
            second_best_idx, second_best_loss = sorted_rt[1]
            rt1_ss, rt2_ss = model.rt_heads[best_idx], model.rt_heads[second_best_idx]
            snapshots.extend([rt1_ss.state_dict(),rt2_ss.state_dict()])
    
    # KD on RT findings (KD doesn't assume linear mode connectivity, which hurt performance. More in the README)
    if len(sorted_rt) > 0:
        first_idx_rt, first_loss_rt = sorted_rt[0]
        
        if 'runway_counter' not in head_patience:
            head_patience['runway_counter'] = 0
            head_patience['best_candidate_sd'] = None
        # If RT loss drops more than 10% add it to C weights
        if first_loss_rt < rt_temp_loss * 0.9: # (20% was overly optimistic)
            head_patience['runway_counter'] += 1
            rt_temp_loss = first_loss_rt
            head_patience['best_candidate_sd'] = model.rt_heads[first_idx_rt].state_dict()
            head_patience['best_candidate_idx'] = first_idx_rt
            tqdm.write(f"RT Head breakthrough found, runway progress: {head_patience['runway_counter']}/3")
        else:
            pass
        
        if head_patience['runway_counter'] >= 3: # If runway is passed
            tqdm.write(f"RT runway cleared, running KD into C heads.")
            rt_sd = head_patience['best_candidate_sd']
            teacher_idx = head_patience['best_candidate_idx']
            teacher_rt = model.rt_heads[teacher_idx]
            teacher_rt.eval()

            model.train()
            for kd_step, (kd_img,kd_lbl) in enumerate(train_loader):
                if kd_step>30: # optimization window, might need adjustment
                    break
                
                kd_img = kd_img.to(device,non_blocking=True)
                kd_lbl = kd_lbl.to(device,non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
            
                with torch.no_grad():
                    teacher_out = teacher_rt(kd_img)
                    teacher_logits = teacher_out[0] if isinstance(teacher_out,tuple) else teacher_out # In case changed in the future
                
                logits_c,features_c, _, _ = model(kd_img)

                kd_loss = gvnr.kd_loss(logits_c,teacher_logits,kd_lbl,criterion_c,alpha=0.5,temperature=4.0)

                kd_loss.backward()
                optimizer.step()

            head_patience['runway_counter'] = 0
            head_patience['best_candidate_sd'] = None
            head_patience['best_candidate_idx'] = None

    # Pruning the bottom 25% every n epochs
    if (epoch+1)%elemination_patience==0:
        rt_count= int(len(model.rt_heads.keys())/4)
        sorted_rt = reversed(sorted_rt)
        indicies,_ = sorted_rt[:rt_count]

        for idx in indicies:
            model.remove_rt(idx)

    scheduler.step()