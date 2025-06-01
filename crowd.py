import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset class
class CrowdCountDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, f"seq_{int(row['image_id']):06d}.jpg")
        image = Image.open(img_path).convert("RGB")
        count = torch.tensor([row['count']], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        return image, count

# Model definition
class ResNetRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.backbone(x)

# Train loop
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for images, counts in tqdm(loader, desc="Training", leave=False):
        images, counts = images.to(device), counts.to(device)
        preds = model(images)
        loss = criterion(preds, counts)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

# Validation loop
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_counts = [], []
    with torch.no_grad():
        for images, counts in tqdm(loader, desc="Validating", leave=False):
            images, counts = images.to(device), counts.to(device)
            preds = model(images)
            loss = criterion(preds, counts)
            total_loss += loss.item() * images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_counts.extend(counts.cpu().numpy())
    return total_loss / len(loader.dataset), all_preds, all_counts

# Main script
if __name__ == "__main__":
    # Load data
    df = pd.read_csv("training_data.csv")
    df.columns = ['image_id', 'count']
    df['image_id'] = df['image_id'].astype(str).str.extract('(\d+)').astype(int)

    train_df, val_df = train_test_split(df, test_size=0.25, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(CrowdCountDataset(train_df, "train", transform),
                              batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(CrowdCountDataset(val_df, "train", val_transform),
                            batch_size=16, shuffle=False, num_workers=2)

    model = ResNetRegressor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.SmoothL1Loss()

    best_val_loss = float('inf')
    for epoch in range(1, 31):
        print(f"\nEpoch {epoch}/30")
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss, val_preds, val_true = validate(model, val_loader, criterion)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Model saved!")

    # Save predictions for analysis
    pred_df = pd.DataFrame({
        "image_id": val_df["image_id"].values,
        "actual_count": np.array(val_true).flatten(),
        "predicted_count": np.array(val_preds).flatten(),
        "accuracy(%)": 100 - (np.abs(np.array(val_preds).flatten() - np.array(val_true).flatten()) / np.array(val_true).flatten()) * 100
    })
    pred_df.to_csv("validation_results.csv", index=False)
    print("\n✅ Validation predictions saved to 'validation_results.csv'")
