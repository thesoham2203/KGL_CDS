

# 1. Imports
import os
import pandas as pd
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import resnet50

# 2. Device & Paths
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

TRAIN_IMG_DIR = '/content/drive/MyDrive/kaggle-olympiad-crowd-density-prediction/output_train/train'
TRAIN_CSV     = '/content/drive/MyDrive/kaggle-olympiad-crowd-density-prediction/training_data.csv'
TEST_IMG_DIR  = '/content/drive/MyDrive/kaggle-olympiad-crowd-density-prediction/test/test'
OUTPUT_SUB    = 'submission_highacc.csv'

# 3. Dataset with Augmentation
class CountDataset(Dataset):
    def __init__(self, img_dir, annotations_df=None, ids_list=None, transform=None):
        self.img_dir = img_dir
        self.ann_df = annotations_df.set_index('id') if annotations_df is not None else None
        self.ids = ids_list or self.ann_df.index.tolist()
        self.transform = transform

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        path = os.path.join(self.img_dir, f'seq_{img_id:06d}.jpg')
        img = Image.open(path).convert('RGB')
        x = self.transform(img)
        if self.ann_df is not None:
            y = torch.tensor([self.ann_df.loc[img_id, 'count']], dtype=torch.float32)
            return x, y
        else:
            return x, img_id

# 4. Transforms
train_tf = T.Compose([
    T.Resize((512, 512)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2, 0.1),
    T.RandomRotation(5),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_tf = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 5. Prepare Data
df = pd.read_csv(TRAIN_CSV)
train_ids = df['id'].iloc[:1000].tolist()
val_ids   = df['id'].iloc[1000:].tolist()
test_ids  = [int(p.split('_')[1].split('.')[0]) for p in sorted(os.listdir(TEST_IMG_DIR)) if p.endswith('.jpg')]

train_ds = CountDataset(TRAIN_IMG_DIR, df, train_ids, transform=train_tf)
val_ds   = CountDataset(TRAIN_IMG_DIR, df, val_ids, transform=val_tf)
test_ds  = CountDataset(TEST_IMG_DIR, annotations_df=None, ids_list=test_ids, transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)

# 6. Better ResNet Model
class ResNetRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet50(pretrained=True)
        for param in base.parameters(): param.requires_grad = False
        for param in base.layer4.parameters(): param.requires_grad = True

        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        return self.regressor(x)

model = ResNetRegressor().to(device)

# 7. Training Setup
criterion = nn.L1Loss()  # MAE is more robust to outliers
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3)
num_epochs = 30

best_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validation"):
            x, y = x.to(device), y.to(device)
            out = model(x)
            val_loss += criterion(out, y).item()

    train_mae = train_loss / len(train_ds)
    val_mae = val_loss / len(val_ds)
    print(f"\n📊 Epoch {epoch+1}: Train MAE = {train_mae:.4f}, Val MAE = {val_mae:.4f}")

    scheduler.step(val_mae)

    if val_mae < best_loss:
        best_loss = val_mae
        torch.save(model.state_dict(), "resnet50_regressor_best.pth")

# 8. Inference
print("\n🔍 Running inference on test set...")
model.load_state_dict(torch.load("resnet50_regressor_best.pth"))
model.eval()
preds = []

with torch.no_grad():
    for x, ids in tqdm(test_loader, desc="Test Inference"):
        x = x.to(device)
        out = model(x).cpu().squeeze().round().int().tolist()
        if isinstance(ids, torch.Tensor): ids = ids.tolist()
        for img_id, cnt in zip(ids, out):
            preds.append({"id": img_id, "count": int(cnt)})

# 9. Save Submission
sub_df = pd.DataFrame(preds).sort_values("id")
sub_df.to_csv(OUTPUT_SUB, index=False)
print(f"✅ Submission saved to `{OUTPUT_SUB}`")
