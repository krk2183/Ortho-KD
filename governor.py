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
    
    def compute_rt_loss(self,rt_features_list, rt_logits_list, targets, criterion):
        """
        rt_features_list: list of pooled feature tensors from each RT head [batch_size, hidden_dim]
        rt_logits_list: list of final logit tensors from each RT head [batch_size, out_dim]
        """
        task_losses = [criterion(logits, targets) for logits in rt_logits_list]
        total_task_loss = sum(task_losses) / len(task_losses)
        
        innovator_feat = rt_features_list[0]
        follower_feats = rt_features_list[1:]
        
        diversity_loss = 0.0
        if len(follower_feats) > 0:
            innovator_norm = F.normalize(innovator_feat, dim=1)
            
            for f_feat in follower_feats:
                f_norm = F.normalize(f_feat, dim=1)
                cosine_sim = (innovator_norm * f_norm).sum(dim=1).mean()
                

                diversity_loss += torch.abs(cosine_sim)
                
            diversity_loss = diversity_loss / len(follower_feats)
            
        lambda_div = 0.1 # how aggressively they must differ
        total_loss = total_task_loss + (lambda_div * diversity_loss)
        
        return total_loss

