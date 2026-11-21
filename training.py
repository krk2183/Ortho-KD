from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from model import MultiHead
from tinyimagenet import TinyImageNet
from governor import Governor

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gvnr = Governor()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_dir = Path("./data/tinyimagenet")

train_dataset = TinyImageNet(data_dir, split="train", transform=transform)
val_dataset = TinyImageNet(data_dir, split="val", transform=transform)

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
        "lr": 5e-5,
        "weight_decay": 1e-2,
    },
    {
        "params": model.rt_heads.parameters(),
        "lr": 1e-3,
        "weight_decay": 1e-3,
    },
])
criterion_c, criterion_rt = (
    nn.CrossEntropyLoss(label_smoothing=0.05),
    nn.CrossEntropyLoss(),
)
patience_limit = 3 # recommended 20% of num epochs
EPOCHS = 15
snapshots = []
best_c_loss = 0.0
rt_temp_loss = 0.0
head_patience = {
    'rt_heads':{name:{"best_loss":float("-inf"), "patience":0} for name in model.rt_heads.keys()}
}
for epoch in range(EPOCHS):
    training_loss_c, training_loss_rt = 0.0, 0.0
    model.train()
    train_pbar = tqdm(train_loader,desc=f"Epoch [{epoch + 1:02d}/{EPOCHS:02d}] Train",leave=False,)

    for step, (image, label) in enumerate(train_pbar, start=1):
        image = image.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits_c, logits_rt = model(image)

        loss_c = sum(criterion_c(head, label) for head in logits_c) / len(logits_c)
        loss_rt = sum(criterion_rt(head, label) for head in logits_rt) / len(logits_rt)

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

            logits_c, logits_rt = model(image)

            loss_c = sum(criterion_c(head, label) for head in logits_c) / len(logits_c)
            loss_rt = sum(criterion_rt(head, label) for head in logits_rt) / len(logits_rt)

            val_loss_c_sum += loss_c.item()
            val_loss_rt_sum += loss_rt.item()

            stacked_logits = torch.stack(logits_c,dim=0)
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
    sample_x,sample_target = val_loader.dataset[torch.randint(0,len(val_loader.dataset),(1,)).item()]
    epoch_metrics = model.eval_head(sample_x,sample_target,criterion_c,criterion_rt)
    model, head_patience = gvnr.clean_model(model,epoch_metrics,head_patience,snapshots,patience_limit=patience_limit)

    # snapshotting
    if (epoch+1)>=int(EPOCHS*0.2):
        c_vals = epoch_metrics["c_heads"].values()
        mean_c_loss = sum(c_vals) / len(c_vals) 
        if mean_c_loss>best_c_loss:
            best_c_loss=mean_c_loss
            avg_weights = model.get_c_avg()
            snapshots.append(avg_weights)

            # Get the best 2 RT heads
            rt_losses = epoch_metrics['rt_heads']
            sorted_rt = sorted(rt_losses.items(),key=lambda item: item[1]) # sort by loss 
            best_idx,second_best_idx = sorted_rt[0][1], sorted_rt[1][1]
            rt1_ss, rt2_ss = model.rt_heads[best_idx], model.rt_heads[second_best_idx]
            snapshots.append(rt1_ss.state_dict(),rt2_ss.state_dict())

    # C heads using RT findings
    # ⚠️⚠️ Temporary ⚠️⚠️ TO DO: Add Governor functionality
    # Temporary logic: If RT loss drops more than 10% add it to C weights
    for head in model.c_heads.values():
        c_sd = head.state_dict()
        first_loss_rt, first_idx_rt = sorted_rt[0]
        if rt_temp_loss*1.1>first_loss_rt:
            combined_weights = 0.65*c_sd+0.35*models.rt_heads[first_idx_rt].state_dict()
            head.load_state_dict(combined_weights)