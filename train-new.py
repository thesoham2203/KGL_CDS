import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import CSRNet
from dataset import CrowdDataset
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = CrowdDataset("training_data.csv", "train")
val_dataset = CrowdDataset("val_labels.csv", "val")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

model = CSRNet().to(device)
criterion = nn.MSELoss()
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

best_val_loss = float('inf')
early_stop_counter = 0

for epoch in range(50):
    model.train()
    train_loss = 0.0
    for imgs, counts in train_loader:
        imgs, counts = imgs.to(device), counts.to(device)
        preds = model(imgs).sum(dim=[2, 3])
        loss = criterion(preds.view(-1), counts)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, counts in val_loader:
            imgs, counts = imgs.to(device), counts.to(device)
            preds = model(imgs).sum(dim=[2, 3])
            loss = criterion(preds.view(-1), counts)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, "
          f"Val Loss = {val_loss/len(val_loader):.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), "best_model_new.pth")
    else:
        early_stop_counter += 1
        if early_stop_counter > 7:
            print("Early stopping.")
            break
