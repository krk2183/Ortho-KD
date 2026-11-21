import torch
import torch.nn as nn
import torch.nn.functional as F

class Governor(nn.Module):
    def __init__(self):
        super().__init__()

    def clean_model(self,model,epoch_metrics,head_patience,snapshots,patience_limit=3):
        for name in model.rt_heads.keys():
            if name not in head_patience['rt_heads']:
                head_patience['rt_heads'][name] = {'best_loss':float('-inf'), "patience":0}
            loss,state = epoch_metrics['rt_heads'][name], head_patience['rt_heads'][name]

            if loss<state['best_loss']:
                state['best_loss'] = loss
                state['patience'] = 0
            else:
                state['patience'] +=1
            
            if state['patience']>=patience_limit:
                model.remove_rt(name)
                del head_patience['rt_heads'][name]
                model.add_rt(snapshots)
        return model, head_patience
    
    def select_rt(self):
        return 0

