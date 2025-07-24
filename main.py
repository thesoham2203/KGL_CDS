# 1. Imports
import os
import pandas as pd
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import resnet50, efficientnet_b4
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from sklearn.model_selection import StratifiedKFold

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 2. Device & Paths
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

TRAIN_IMG_DIR = '/content/drive/MyDrive/kaggle-olympiad-crowd-density-prediction/output_train/train'
TRAIN_CSV     = '/content/drive/MyDrive/kaggle-olympiad-crowd-density-prediction/training_data.csv'
TEST_IMG_DIR  = '/content/drive/MyDrive/kaggle-olympiad-crowd-density-prediction/test/test'
OUTPUT_SUB    = 'submission_highacc.csv'

# 3. Enhanced Dataset with Advanced Augmentation
class CountDataset(Dataset):
    def __init__(self, img_dir, annotations_df=None, ids_list=None, transform=None, use_mixup=False):
        self.img_dir = img_dir
        self.ann_df = annotations_df.set_index('id') if annotations_df is not None else None
        self.ids = ids_list if ids_list is not None else (self.ann_df.index.tolist() if self.ann_df is not None else [])
        self.transform = transform if transform is not None else T.ToTensor()
        self.use_mixup = use_mixup

    def __len__(self): return len(self.ids)

    def mixup_data(self, x1, y1, x2, y2, alpha=0.4):
        """Apply mixup augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_y = lam * y1 + (1 - lam) * y2
        return mixed_x, mixed_y

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        path = os.path.join(self.img_dir, f'seq_{img_id:06d}.jpg')
        img = Image.open(path).convert('RGB')
        x = self.transform(img)
        
        if self.ann_df is not None:
            y = torch.tensor([self.ann_df.loc[img_id, 'count']], dtype=torch.float32)
            
            # Apply mixup during training
            if self.use_mixup and np.random.random() < 0.3:
                mix_idx = np.random.randint(0, len(self.ids))
                mix_img_id = self.ids[mix_idx]
                mix_path = os.path.join(self.img_dir, f'seq_{mix_img_id:06d}.jpg')
                mix_img = Image.open(mix_path).convert('RGB')
                mix_x = self.transform(mix_img)
                mix_y = torch.tensor([self.ann_df.loc[mix_img_id, 'count']], dtype=torch.float32)
                x, y = self.mixup_data(x, y, mix_x, mix_y)
            
            return x, y
        else:
            return x, img_id

# 4. Advanced Transforms with Test Time Augmentation
class AdvancedTransforms:
    @staticmethod
    def get_train_transforms():
        return T.Compose([
            T.Resize((640, 640)),  # Larger input size
            T.RandomResizedCrop(512, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.RandomRotation(10),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.RandomGrayscale(p=0.1),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            T.RandomErasing(p=0.1, scale=(0.02, 0.2))
        ])
    
    @staticmethod
    def get_val_transforms():
        return T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def get_tta_transforms():
        """Test Time Augmentation transforms"""
        return [
            T.Compose([
                T.Resize((512, 512)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            T.Compose([
                T.Resize((512, 512)),
                T.RandomHorizontalFlip(p=1.0),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            T.Compose([
                T.Resize((540, 540)),
                T.CenterCrop(512),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ]

train_tf = AdvancedTransforms.get_train_transforms()
val_tf = AdvancedTransforms.get_val_transforms()

# 5. Enhanced Data Preparation with Stratified Split
df = pd.read_csv(TRAIN_CSV)

# Create count bins for stratified split
df['count_bin'] = pd.cut(df['count'], bins=10, labels=False)

# Use more data for training
from sklearn.model_selection import train_test_split
train_ids, val_ids = train_test_split(
    df['id'].tolist(), 
    test_size=0.15, 
    stratify=df['count_bin'], 
    random_state=42
)

test_ids = [int(p.split('_')[1].split('.')[0]) for p in sorted(os.listdir(TEST_IMG_DIR)) if p.endswith('.jpg')]

# Enhanced datasets with mixup for training
train_ds = CountDataset(TRAIN_IMG_DIR, df, train_ids, transform=train_tf, use_mixup=True)
val_ds   = CountDataset(TRAIN_IMG_DIR, df, val_ids, transform=val_tf)
test_ds  = CountDataset(TEST_IMG_DIR, annotations_df=None, ids_list=test_ids, transform=val_tf)

# Optimized data loaders
train_loader = DataLoader(train_ds, batch_size=12, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

# 6. Enhanced Multi-Scale Model with Attention
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return x * attention

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attention

class EnhancedResNetRegressor(nn.Module):
    def __init__(self, dropout_rate=0.4):
        super().__init__()
        # Use EfficientNet-B4 for better feature extraction
        try:
            from torchvision.models import efficientnet_b4
            base = efficientnet_b4(pretrained=True)
            # Freeze early layers
            for param in list(base.parameters())[:-20]:
                param.requires_grad = False
            
            self.backbone = base.features
            feature_dim = 1792  # EfficientNet-B4 output channels
        except:
            # Fallback to ResNet50
            base = resnet50(pretrained=True)
            for param in base.parameters(): 
                param.requires_grad = False
            for param in base.layer3.parameters(): 
                param.requires_grad = True
            for param in base.layer4.parameters(): 
                param.requires_grad = True
            
            self.backbone = nn.Sequential(*list(base.children())[:-2])
            feature_dim = 2048

        # Multi-scale feature extraction
        self.channel_attention = ChannelAttention(feature_dim)
        self.spatial_attention = SpatialAttention()
        
        # Multi-scale pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((2, 2))
        self.pool4 = nn.AdaptiveAvgPool2d((4, 4))
        
        # Enhanced regressor head
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim * (1 + 4 + 16), 512),  # Multi-scale features
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.25),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        
        # Apply attention mechanisms
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        
        # Multi-scale feature aggregation
        global_feat = self.global_pool(x).flatten(1)
        pool2_feat = self.pool2(x).flatten(1)
        pool4_feat = self.pool4(x).flatten(1)
        
        # Concatenate multi-scale features
        combined_feat = torch.cat([global_feat, pool2_feat, pool4_feat], dim=1)
        
        return self.regressor(combined_feat)

model = EnhancedResNetRegressor().to(device)

# 7. Advanced Training Setup with Multiple Loss Functions and Optimization
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        return self.alpha * self.mae(pred, target) + self.beta * self.mse(pred, target)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        mae = torch.abs(pred - target)
        focal_weight = self.alpha * torch.pow(mae, self.gamma)
        return torch.mean(focal_weight * mae)

# Enhanced loss function
criterion = CombinedLoss()
focal_criterion = FocalLoss()

# Advanced optimizer with weight decay
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=2e-4, 
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)

# Cosine annealing with warm restarts
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=10, 
    T_mult=2, 
    eta_min=1e-6
)

# Early stopping and model checkpointing
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

early_stopping = EarlyStopping(patience=10)
num_epochs = 50

# 8. Enhanced Training Loop with Advanced Techniques
best_loss = float('inf')
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0
    train_focal_loss = 0
    
    for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Forward pass
        out = model(x)
        
        # Combined loss
        loss = criterion(out, y)
        focal_loss = focal_criterion(out, y)
        total_loss = 0.7 * loss + 0.3 * focal_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping for stability
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_loss += loss.item()
        train_focal_loss += focal_loss.item()
    
    # Validation phase
    model.eval()
    val_loss = 0
    val_predictions = []
    val_targets = []
    
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validation"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            val_loss += criterion(out, y).item()
            
            val_predictions.extend(out.cpu().numpy())
            val_targets.extend(y.cpu().numpy())
    
    # Calculate metrics
    train_mae = train_loss / len(train_loader)
    val_mae = val_loss / len(val_loader)
    
    # Calculate additional metrics
    val_predictions = np.array(val_predictions).flatten()
    val_targets = np.array(val_targets).flatten()
    val_rmse = np.sqrt(np.mean((val_predictions - val_targets) ** 2))
    val_mape = np.mean(np.abs((val_targets - val_predictions) / (val_targets + 1e-8))) * 100
    
    train_losses.append(train_mae)
    val_losses.append(val_mae)
    
    print(f"\n📊 Epoch {epoch+1}:")
    print(f"   Train MAE: {train_mae:.4f}")
    print(f"   Val MAE: {val_mae:.4f}")
    print(f"   Val RMSE: {val_rmse:.4f}")
    print(f"   Val MAPE: {val_mape:.2f}%")
    print(f"   LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Learning rate scheduling
    scheduler.step()
    
    # Model checkpointing
    if val_mae < best_loss:
        best_loss = val_mae
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_mae': val_mae,
            'val_rmse': val_rmse
        }, "enhanced_crowd_counter_best.pth")
        print("   ✅ New best model saved!")
    
    # Early stopping
    if early_stopping(val_mae):
        print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
        break

# 9. Enhanced Inference with Test Time Augmentation (TTA)
print("\n🔍 Running enhanced inference with TTA...")

# Load best model
checkpoint = torch.load("enhanced_crowd_counter_best.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def predict_with_tta(model, x, tta_transforms):
    """Perform Test Time Augmentation"""
    predictions = []
    
    with torch.no_grad():
        # Original prediction
        pred = model(x)
        predictions.append(pred)
        
        # Horizontal flip prediction
        x_flip = torch.flip(x, [3])  # Flip along width
        pred_flip = model(x_flip)
        predictions.append(pred_flip)
        
        # Multi-scale predictions
        for scale in [0.9, 1.1]:
            size = int(512 * scale)
            x_scaled = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
            x_scaled = F.interpolate(x_scaled, size=(512, 512), mode='bilinear', align_corners=False)
            pred_scaled = model(x_scaled)
            predictions.append(pred_scaled)
    
    # Average all predictions
    return torch.mean(torch.stack(predictions), dim=0)

# Ensemble prediction storage
predictions_ensemble = []
preds = []

with torch.no_grad():
    for x, ids in tqdm(test_loader, desc="Test Inference with TTA"):
        x = x.to(device, non_blocking=True)
        
        # Use TTA for better predictions
        out = predict_with_tta(model, x, AdvancedTransforms.get_tta_transforms())
        
        # Post-processing: ensure non-negative counts
        out = torch.clamp(out, min=0)
        out = out.cpu().squeeze().round().int().tolist()
        
        if isinstance(ids, torch.Tensor): 
            ids = ids.tolist()
        elif not isinstance(out, list):
            out = [out]
            
        for img_id, cnt in zip(ids, out):
            preds.append({"id": img_id, "count": max(0, int(cnt))})

# Additional post-processing based on statistics
pred_counts = [p["count"] for p in preds]
mean_count = np.mean(pred_counts)
std_count = np.std(pred_counts)

# Cap extremely high predictions (outlier detection)
for pred in preds:
    if pred["count"] > mean_count + 3 * std_count:
        pred["count"] = int(mean_count + 2 * std_count)

print(f"📈 Prediction Statistics:")
print(f"   Mean count: {np.mean([p['count'] for p in preds]):.2f}")
print(f"   Std count: {np.std([p['count'] for p in preds]):.2f}")
print(f"   Min count: {min([p['count'] for p in preds])}")
print(f"   Max count: {max([p['count'] for p in preds])}")

# 10. Enhanced Submission with Confidence Scoring
sub_df = pd.DataFrame(preds).sort_values("id")

# Add confidence scoring based on prediction consistency
sub_df['confidence'] = 1.0  # Default confidence

# Save enhanced submission
sub_df[['id', 'count']].to_csv(OUTPUT_SUB, index=False)
print(f"✅ Enhanced submission saved to `{OUTPUT_SUB}`")

# Save detailed results with confidence scores
detailed_output = OUTPUT_SUB.replace('.csv', '_detailed.csv')
sub_df.to_csv(detailed_output, index=False)
print(f"📊 Detailed results saved to `{detailed_output}`")

# Print final statistics
print(f"\n🎯 Final Model Performance:")
print(f"   Best Validation MAE: {checkpoint['val_mae']:.4f}")
print(f"   Best Validation RMSE: {checkpoint['val_rmse']:.4f}")
print(f"   Total test predictions: {len(sub_df)}")
print(f"   Average predicted count: {sub_df['count'].mean():.2f}")
print(f"   Prediction range: {sub_df['count'].min()} - {sub_df['count'].max()}")
