import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import CrowdDataset

class CountRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet18(weights='IMAGENET1K_V1')
        self.base.fc = nn.Linear(self.base.fc.in_features, 1)

    def forward(self, x):
        return self.base(x)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = CrowdDataset("training_data.csv", "train", transform=train_transform)
    val_dataset   = CrowdDataset("val_labels.csv", "val", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    model = CountRegressor().to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    patience = 0

    for epoch in range(1, 80):
        model.train()
        train_loss = 0.0
        for imgs, counts in train_loader:
            imgs, counts = imgs.to(device), counts.to(device)
            preds = model(imgs).view(-1)
            loss = criterion(preds, counts)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        avg_train = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, counts in val_loader:
                imgs, counts = imgs.to(device), counts.to(device)
                preds = model(imgs).view(-1)
                loss = criterion(preds, counts)
                val_loss += loss.item() * imgs.size(0)

        avg_val = val_loss / len(val_loader.dataset)
        scheduler.step(val_loss)


        print(f"Epoch {epoch}: Train Loss = {avg_train:.4f}, Val Loss = {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience = 0
            torch.save(model.state_dict(), "regressor_best_model.pth")
            print("✅ Model saved!")
        else:
            patience += 1
            if patience > 7:
                print("⏹️ Early stopping triggered.")
                break

    print("🎉 Training complete.")
