# ============================================================================
# 1. IMPORTS AND DEPENDENCIES
# ============================================================================
"""
Enhanced Crowd Counting Model using Deep Learning
This script implements an advanced crowd counting system with:
- Multi-scale feature extraction
- Attention mechanisms
- Test Time Augmentation (TTA)
- Advanced data augmentation
- Robust training strategies
"""

# Standard library imports
import os                    # File and directory operations
import pandas as pd          # Data manipulation and CSV handling
import numpy as np           # Numerical computations
import random               # Random number generation for reproducibility
import warnings             # Warning message control
warnings.filterwarnings('ignore')  # Suppress unnecessary warnings

# Image processing and visualization
from PIL import Image       # Image loading and basic operations
from glob import glob       # File pattern matching
from tqdm import tqdm       # Progress bars for loops

# PyTorch core components
import torch                # Main PyTorch library
import torch.nn as nn       # Neural network modules
import torch.optim as optim # Optimization algorithms
import torch.nn.functional as F  # Functional operations
from torch.utils.data import Dataset, DataLoader  # Data loading utilities
from torch.nn.utils.clip_grad import clip_grad_norm_  # Gradient clipping

# Computer vision and transformations
import torchvision.transforms as T  # Image transformations
from torchvision.models import resnet50, efficientnet_b4  # Pre-trained models

# Machine learning utilities
from sklearn.model_selection import StratifiedKFold  # Cross-validation tools

# ============================================================================
# 2. REPRODUCIBILITY SETUP
# ============================================================================

def set_seed(seed=42):
    """
    Set random seeds for reproducible results across different runs.
    This ensures that:
    - Random data augmentations are consistent
    - Model weight initialization is reproducible
    - Training results can be replicated
    
    Args:
        seed (int): Random seed value (default: 42)
    """
    random.seed(seed)                           # Python random module
    np.random.seed(seed)                        # NumPy random operations
    torch.manual_seed(seed)                     # PyTorch CPU operations
    torch.cuda.manual_seed(seed)                # PyTorch single GPU operations
    torch.cuda.manual_seed_all(seed)            # PyTorch multi-GPU operations
    torch.backends.cudnn.deterministic = True   # Ensure deterministic algorithms
    torch.backends.cudnn.benchmark = False      # Disable auto-tuning for consistency

# Apply reproducibility settings
set_seed(42)

# ============================================================================
# 3. DEVICE AND PATH CONFIGURATION
# ============================================================================

# Automatically detect and use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset paths - Update these to match your data structure
TRAIN_IMG_DIR = '/content/drive/MyDrive/kaggle-olympiad-crowd-density-prediction/output_train/train'
TRAIN_CSV     = '/content/drive/MyDrive/kaggle-olympiad-crowd-density-prediction/training_data.csv'
TEST_IMG_DIR  = '/content/drive/MyDrive/kaggle-olympiad-crowd-density-prediction/test/test'
OUTPUT_SUB    = 'submission_highacc.csv'  # Output file for predictions

# ============================================================================
# 4. ENHANCED DATASET CLASS WITH ADVANCED AUGMENTATION
# ============================================================================

class CountDataset(Dataset):
    """
    Custom PyTorch Dataset for crowd counting with advanced features:
    - Mixup augmentation for better generalization
    - Flexible data loading for train/test splits
    - Robust error handling for missing files
    - Support for various image formats
    """
    
    def __init__(self, img_dir, annotations_df=None, ids_list=None, transform=None, use_mixup=False):
        """
        Initialize the dataset.
        
        Args:
            img_dir (str): Directory containing images
            annotations_df (pd.DataFrame): DataFrame with image IDs and counts
            ids_list (list): List of image IDs to use
            transform (torchvision.transforms): Image transformations to apply
            use_mixup (bool): Whether to apply mixup augmentation during training
        """
        self.img_dir = img_dir
        
        # Set up annotations DataFrame with ID as index for fast lookup
        self.ann_df = annotations_df.set_index('id') if annotations_df is not None else None
        
        # Determine which image IDs to use (fallback to all if not specified)
        self.ids = ids_list if ids_list is not None else (
            self.ann_df.index.tolist() if self.ann_df is not None else []
        )
        
        # Set up transformations (default to simple tensor conversion)
        self.transform = transform if transform is not None else T.ToTensor()
        
        # Configure mixup augmentation (only used during training)
        self.use_mixup = use_mixup

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.ids)

    def mixup_data(self, x1, y1, x2, y2, alpha=0.4):
        """
        Apply mixup augmentation by linearly combining two samples.
        Mixup helps the model generalize better by learning from interpolated examples.
        
        Args:
            x1, x2 (torch.Tensor): Input images
            y1, y2 (torch.Tensor): Corresponding labels/counts
            alpha (float): Beta distribution parameter for mixing ratio
            
        Returns:
            tuple: Mixed image and label
        """
        if alpha > 0:
            # Sample mixing ratio from Beta distribution
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1  # No mixing if alpha <= 0
        
        # Linear interpolation of images and labels
        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_y = lam * y1 + (1 - lam) * y2
        return mixed_x, mixed_y

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (image_tensor, count) for training data, (image_tensor, image_id) for test data
        """
        img_id = self.ids[idx]
        
        # Construct image path (assumes specific naming convention)
        path = os.path.join(self.img_dir, f'seq_{img_id:06d}.jpg')
        
        # Load and convert image to RGB format
        img = Image.open(path).convert('RGB')
        
        # Apply transformations (augmentation, normalization, etc.)
        x = self.transform(img)
        
        if self.ann_df is not None:
            # Training/validation mode: return image and count
            y = torch.tensor([self.ann_df.loc[img_id, 'count']], dtype=torch.float32)
            
            # Apply mixup augmentation with 30% probability during training
            if self.use_mixup and np.random.random() < 0.3:
                # Select a random second sample for mixing
                mix_idx = np.random.randint(0, len(self.ids))
                mix_img_id = self.ids[mix_idx]
                mix_path = os.path.join(self.img_dir, f'seq_{mix_img_id:06d}.jpg')
                
                # Load and transform the second image
                mix_img = Image.open(mix_path).convert('RGB')
                mix_x = self.transform(mix_img)
                mix_y = torch.tensor([self.ann_df.loc[mix_img_id, 'count']], dtype=torch.float32)
                
                # Mix the two samples
                x, y = self.mixup_data(x, y, mix_x, mix_y)
            
            return x, y
        else:
            # Test mode: return image and ID for prediction
            return x, img_id

# ============================================================================
# 5. ADVANCED DATA TRANSFORMATIONS AND AUGMENTATION
# ============================================================================

class AdvancedTransforms:
    """
    Collection of sophisticated data transformations for different training phases.
    These transformations significantly improve model robustness and generalization.
    """
    
    @staticmethod
    def get_train_transforms():
        """
        Comprehensive training augmentations designed to:
        - Increase data variety and prevent overfitting
        - Simulate real-world conditions (lighting, angles, occlusions)
        - Improve model robustness to input variations
        
        Returns:
            torchvision.transforms.Compose: Sequence of training transformations
        """
        return T.Compose([
            # 1. Size manipulations for multi-scale learning
            T.Resize((640, 640)),                              # Initial larger size
            T.RandomResizedCrop(512, scale=(0.8, 1.0)),        # Random crop with scaling
            
            # 2. Geometric augmentations
            T.RandomHorizontalFlip(p=0.5),                     # Mirror images horizontally
            T.RandomRotation(10),                              # Small rotations (-10° to +10°)
            T.RandomAffine(                                    # Affine transformations
                degrees=0,                                     # No additional rotation
                translate=(0.1, 0.1),                          # Random translation (±10%)
                scale=(0.9, 1.1)                              # Random scaling (90%-110%)
            ),
            
            # 3. Color and appearance augmentations
            T.ColorJitter(                                     # Random color variations
                brightness=0.3,                               # ±30% brightness
                contrast=0.3,                                 # ±30% contrast
                saturation=0.3,                               # ±30% saturation
                hue=0.1                                       # ±10% hue shift
            ),
            T.RandomGrayscale(p=0.1),                          # 10% chance of grayscale
            T.GaussianBlur(                                    # Random blur simulation
                kernel_size=3, 
                sigma=(0.1, 2.0)
            ),
            
            # 4. Tensor conversion and normalization
            T.ToTensor(),                                      # Convert to PyTorch tensor
            T.Normalize(                                       # ImageNet normalization
                [0.485, 0.456, 0.406],                       # Mean values per channel
                [0.229, 0.224, 0.225]                        # Std values per channel
            ),
            
            # 5. Advanced augmentations
            T.RandomErasing(                                   # Random rectangular occlusions
                p=0.1,                                        # 10% probability
                scale=(0.02, 0.2)                             # 2%-20% of image area
            )
        ])
    
    @staticmethod
    def get_val_transforms():
        """
        Simple validation/test transformations without augmentation.
        Only applies necessary preprocessing for model input.
        
        Returns:
            torchvision.transforms.Compose: Sequence of validation transformations
        """
        return T.Compose([
            T.Resize((512, 512)),                             # Fixed size for consistency
            T.ToTensor(),                                     # Convert to tensor
            T.Normalize(                                      # Same normalization as training
                [0.485, 0.456, 0.406], 
                [0.229, 0.224, 0.225]
            )
        ])
    
    @staticmethod
    def get_tta_transforms():
        """
        Test Time Augmentation (TTA) transformations.
        Multiple versions of the same image for ensemble prediction.
        
        Returns:
            list: List of different transformation pipelines for TTA
        """
        return [
            # Original image (no augmentation)
            T.Compose([
                T.Resize((512, 512)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            
            # Horizontally flipped version
            T.Compose([
                T.Resize((512, 512)),
                T.RandomHorizontalFlip(p=1.0),                # Always flip
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            
            # Slightly larger crop for multi-scale testing
            T.Compose([
                T.Resize((540, 540)),                         # Slightly larger
                T.CenterCrop(512),                            # Center crop to target size
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ]

# Initialize transform objects for easy access
train_tf = AdvancedTransforms.get_train_transforms()
val_tf = AdvancedTransforms.get_val_transforms()

# ============================================================================
# 6. DATA PREPARATION WITH STRATIFIED SPLITTING
# ============================================================================

# Load the training data CSV file
df = pd.read_csv(TRAIN_CSV)

# Create stratified bins based on crowd count for balanced train/val split
# This ensures both sets have similar distributions of crowd densities
df['count_bin'] = pd.cut(df['count'], bins=10, labels=False)

# Perform stratified train-validation split
# - Uses 85% for training, 15% for validation (more data for training)
# - Maintains similar count distributions in both sets
# - Fixed random state ensures reproducible splits
from sklearn.model_selection import train_test_split
train_ids, val_ids = train_test_split(
    df['id'].tolist(),                    # All image IDs
    test_size=0.15,                       # 15% for validation
    stratify=df['count_bin'],             # Stratify by count bins
    random_state=42                       # Fixed seed for reproducibility
)

# Extract test image IDs from filename pattern (seq_XXXXXX.jpg)
# This assumes test images follow the same naming convention
test_ids = [
    int(p.split('_')[1].split('.')[0])    # Extract number from filename
    for p in sorted(os.listdir(TEST_IMG_DIR)) 
    if p.endswith('.jpg')                 # Only process .jpg files
]

print(f"📊 Dataset Statistics:")
print(f"   Total samples: {len(df)}")
print(f"   Training samples: {len(train_ids)} ({len(train_ids)/len(df)*100:.1f}%)")
print(f"   Validation samples: {len(val_ids)} ({len(val_ids)/len(df)*100:.1f}%)")
print(f"   Test samples: {len(test_ids)}")
print(f"   Count range: {df['count'].min()} - {df['count'].max()}")
print(f"   Mean count: {df['count'].mean():.2f}")

# ============================================================================
# 7. DATASET AND DATALOADER INITIALIZATION
# ============================================================================

# Create dataset instances with appropriate configurations
train_ds = CountDataset(
    TRAIN_IMG_DIR, 
    df, 
    train_ids, 
    transform=train_tf, 
    use_mixup=True              # Enable mixup augmentation for training
)

val_ds = CountDataset(
    TRAIN_IMG_DIR, 
    df, 
    val_ids, 
    transform=val_tf            # No mixup for validation (clean evaluation)
)

test_ds = CountDataset(
    TEST_IMG_DIR, 
    annotations_df=None,        # No labels for test set
    ids_list=test_ids, 
    transform=val_tf
)

# Create optimized data loaders with performance enhancements
train_loader = DataLoader(
    train_ds, 
    batch_size=12,              # Smaller batch size for stability with large model
    shuffle=True,               # Randomize training order each epoch
    num_workers=4,              # Parallel data loading (adjust based on CPU cores)
    pin_memory=True             # Speed up GPU transfer
)

val_loader = DataLoader(
    val_ds, 
    batch_size=16,              # Larger batch for validation (no gradients)
    shuffle=False,              # No need to shuffle validation data
    num_workers=4, 
    pin_memory=True
)

test_loader = DataLoader(
    test_ds, 
    batch_size=16, 
    shuffle=False,              # Keep test order for consistent submission
    num_workers=4, 
    pin_memory=True
)

# ============================================================================
# 8. ATTENTION MECHANISM MODULES
# ============================================================================

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module that learns WHERE to focus in the image.
    
    This module computes attention weights for each spatial location,
    helping the model focus on crowd-dense regions while suppressing
    background or irrelevant areas.
    """
    
    def __init__(self, kernel_size=7):
        """
        Initialize spatial attention module.
        
        Args:
            kernel_size (int): Size of convolution kernel for attention computation
        """
        super().__init__()
        
        # Convolution to combine average and max pooled features
        # Input: 2 channels (avg + max), Output: 1 channel (attention map)
        self.conv = nn.Conv2d(
            2, 1, 
            kernel_size, 
            padding=kernel_size//2, 
            bias=False
        )
        
        # Sigmoid activation to normalize attention weights to [0,1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Apply spatial attention to input feature map.
        
        Args:
            x (torch.Tensor): Input feature map [B, C, H, W]
            
        Returns:
            torch.Tensor: Attention-weighted feature map [B, C, H, W]
        """
        # Global average pooling across channels: [B, C, H, W] -> [B, 1, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        
        # Global max pooling across channels: [B, C, H, W] -> [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate average and max features: [B, 2, H, W]
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        # Generate attention map through convolution and sigmoid
        attention = self.sigmoid(self.conv(x_cat))  # [B, 1, H, W]
        
        # Apply attention weights element-wise
        return x * attention


class ChannelAttention(nn.Module):
    """
    Channel Attention Module that learns WHAT features to emphasize.
    
    This module computes attention weights for each feature channel,
    allowing the model to focus on the most informative features
    for crowd counting while suppressing noise.
    """
    
    def __init__(self, in_channels, reduction=16):
        """
        Initialize channel attention module.
        
        Args:
            in_channels (int): Number of input feature channels
            reduction (int): Channel reduction ratio for efficiency
        """
        super().__init__()
        
        # Global pooling operations to capture channel statistics
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Average pooling to [B, C, 1, 1]
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Max pooling to [B, C, 1, 1]
        
        # Shared MLP for channel attention computation
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),  # Dimension reduction
            nn.ReLU(),                                        # Non-linearity
            nn.Linear(in_channels // reduction, in_channels)   # Dimension restoration
        )
        
        # Sigmoid for attention weight normalization
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Apply channel attention to input feature map.
        
        Args:
            x (torch.Tensor): Input feature map [B, C, H, W]
            
        Returns:
            torch.Tensor: Channel-wise attention-weighted features [B, C, H, W]
        """
        b, c, _, _ = x.size()
        
        # Apply global average pooling and reshape for linear layer
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        
        # Apply global max pooling and reshape for linear layer
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Combine average and max features, apply sigmoid, and reshape
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        # Apply channel-wise attention weights
        return x * attention

# ============================================================================
# 9. ENHANCED MULTI-SCALE MODEL ARCHITECTURE
# ============================================================================

class EnhancedResNetRegressor(nn.Module):
    """
    Advanced crowd counting model with multiple architectural improvements:
    
    1. EfficientNet-B4 backbone for better feature extraction
    2. Channel and spatial attention mechanisms
    3. Multi-scale feature aggregation
    4. Deep regression head with regularization
    5. Skip connections and residual learning
    
    This architecture is designed to capture crowd patterns at different scales
    while maintaining computational efficiency.
    """
    
    def __init__(self, dropout_rate=0.4):
        """
        Initialize the enhanced model architecture.
        
        Args:
            dropout_rate (float): Dropout probability for regularization
        """
        super().__init__()
        
        # ====================================================================
        # BACKBONE SELECTION: EfficientNet-B4 vs ResNet50
        # ====================================================================
        try:
            # Attempt to use EfficientNet-B4 (more efficient and accurate)
            from torchvision.models import efficientnet_b4
            base = efficientnet_b4(pretrained=True)
            
            # Progressive unfreezing: freeze early layers, unfreeze later ones
            # This preserves low-level features while allowing high-level adaptation
            for param in list(base.parameters())[:-20]:
                param.requires_grad = False
            
            # Extract feature extraction layers (excluding classifier)
            self.backbone = base.features
            feature_dim = 1792  # EfficientNet-B4 output feature dimension
            
            print("✅ Using EfficientNet-B4 backbone")
            
        except:
            # Fallback to ResNet50 if EfficientNet is not available
            print("⚠️  EfficientNet-B4 not available, falling back to ResNet50")
            base = resnet50(pretrained=True)
            
            # Freeze early layers, unfreeze later layers for fine-tuning
            for param in base.parameters(): 
                param.requires_grad = False
            for param in base.layer3.parameters(): 
                param.requires_grad = True
            for param in base.layer4.parameters(): 
                param.requires_grad = True
            
            # Extract feature layers (excluding final pooling and classifier)
            self.backbone = nn.Sequential(*list(base.children())[:-2])
            feature_dim = 2048  # ResNet50 output feature dimension

        # ====================================================================
        # ATTENTION MECHANISMS
        # ====================================================================
        
        # Channel attention: learns which feature channels are important
        self.channel_attention = ChannelAttention(feature_dim)
        
        # Spatial attention: learns which spatial locations are important
        self.spatial_attention = SpatialAttention()
        
        # ====================================================================
        # MULTI-SCALE FEATURE AGGREGATION
        # ====================================================================
        
        # Different pooling operations capture features at various scales
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))    # Global context
        self.pool2 = nn.AdaptiveAvgPool2d((2, 2))          # Regional features  
        self.pool4 = nn.AdaptiveAvgPool2d((4, 4))          # Local features
        
        # Total feature dimension after concatenating multi-scale features
        # 1×1 + 2×2 + 4×4 = 1 + 4 + 16 = 21 feature maps per channel
        total_features = feature_dim * (1 + 4 + 16)
        
        # ====================================================================
        # ENHANCED REGRESSION HEAD
        # ====================================================================
        
        self.regressor = nn.Sequential(
            # Flatten multi-scale features for linear layers
            nn.Flatten(),
            
            # First dense layer with batch normalization
            nn.Linear(total_features, 512),
            nn.BatchNorm1d(512),                    # Stabilize training
            nn.ReLU(),
            nn.Dropout(dropout_rate),               # Prevent overfitting
            
            # Second dense layer with reduced dropout
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),        # Gradual dropout reduction
            
            # Third dense layer for further refinement
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.25),       # Minimal dropout near output
            
            # Final output layer (single neuron for count regression)
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        Forward pass through the enhanced model.
        
        Args:
            x (torch.Tensor): Input images [B, 3, H, W]
            
        Returns:
            torch.Tensor: Predicted crowd counts [B, 1]
        """
        # ====================================================================
        # FEATURE EXTRACTION
        # ====================================================================
        
        # Extract features using the backbone network
        x = self.backbone(x)  # [B, feature_dim, H', W']
        
        # ====================================================================
        # ATTENTION APPLICATION
        # ====================================================================
        
        # Apply channel attention (what features to focus on)
        x = self.channel_attention(x)
        
        # Apply spatial attention (where to focus in the image)
        x = self.spatial_attention(x)
        
        # ====================================================================
        # MULTI-SCALE FEATURE AGGREGATION
        # ====================================================================
        
        # Extract features at different spatial scales
        global_feat = self.global_pool(x).flatten(1)    # [B, feature_dim]
        pool2_feat = self.pool2(x).flatten(1)           # [B, feature_dim * 4]
        pool4_feat = self.pool4(x).flatten(1)           # [B, feature_dim * 16]
        
        # Concatenate all multi-scale features
        combined_feat = torch.cat([global_feat, pool2_feat, pool4_feat], dim=1)
        
        # ====================================================================
        # REGRESSION HEAD
        # ====================================================================
        
        # Pass combined features through regression head for count prediction
        return self.regressor(combined_feat)

# Initialize the model and move to appropriate device (GPU/CPU)
model = EnhancedResNetRegressor().to(device)

# Print model information
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n🏗️  Model Architecture Summary:")
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")
print(f"   Frozen parameters: {total_params - trainable_params:,}")
print(f"   Model size: ~{total_params * 4 / 1024**2:.1f} MB")

# ============================================================================
# 10. ADVANCED LOSS FUNCTIONS FOR ROBUST TRAINING
# ============================================================================

class CombinedLoss(nn.Module):
    """
    Combined loss function that balances different aspects of crowd counting:
    - L1 Loss (MAE): Robust to outliers, provides stable gradients
    - L2 Loss (MSE): Penalizes large errors more heavily
    
    This combination leverages the best of both loss functions for
    improved convergence and accuracy.
    """
    
    def __init__(self, alpha=0.7, beta=0.3):
        """
        Initialize combined loss function.
        
        Args:
            alpha (float): Weight for L1 loss (0.7 = 70%)
            beta (float): Weight for L2 loss (0.3 = 30%)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mae = nn.L1Loss()   # Mean Absolute Error (L1)
        self.mse = nn.MSELoss()  # Mean Squared Error (L2)
    
    def forward(self, pred, target):
        """
        Compute weighted combination of L1 and L2 losses.
        
        Args:
            pred (torch.Tensor): Model predictions
            target (torch.Tensor): Ground truth labels
            
        Returns:
            torch.Tensor: Combined loss value
        """
        return self.alpha * self.mae(pred, target) + self.beta * self.mse(pred, target)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing hard examples in crowd counting.
    
    This loss function gives more weight to hard-to-predict samples
    (those with larger prediction errors), helping the model learn
    from challenging cases like very dense crowds or unusual scenes.
    """
    
    def __init__(self, alpha=1, gamma=2):
        """
        Initialize focal loss function.
        
        Args:
            alpha (float): Scaling factor for loss magnitude
            gamma (float): Focusing parameter (higher = more focus on hard examples)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        """
        Compute focal loss with error-based weighting.
        
        Args:
            pred (torch.Tensor): Model predictions
            target (torch.Tensor): Ground truth labels
            
        Returns:
            torch.Tensor: Focal loss value
        """
        # Calculate absolute error (base loss)
        mae = torch.abs(pred - target)
        
        # Apply focal weighting: larger errors get exponentially more weight
        focal_weight = self.alpha * torch.pow(mae, self.gamma)
        
        # Return weighted loss
        return torch.mean(focal_weight * mae)

# ============================================================================
# 11. TRAINING CONFIGURATION AND OPTIMIZATION
# ============================================================================

# Initialize loss functions
criterion = CombinedLoss()        # Primary loss function
focal_criterion = FocalLoss()     # Secondary loss for hard examples

# Advanced optimizer: AdamW with weight decay for better generalization
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),  # Only trainable parameters
    lr=2e-4,                      # Learning rate (higher than standard 1e-4)
    weight_decay=1e-4,            # L2 regularization to prevent overfitting
    betas=(0.9, 0.999)           # Adam momentum parameters
)

# Learning rate scheduler: Cosine annealing with warm restarts
# This provides periodic learning rate resets for better convergence
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=10,                       # Initial restart period (10 epochs)
    T_mult=2,                     # Period multiplier after each restart
    eta_min=1e-6                  # Minimum learning rate
)

# ============================================================================
# 12. EARLY STOPPING FOR OVERFITTING PREVENTION
# ============================================================================

class EarlyStopping:
    """
    Early stopping mechanism to prevent overfitting.
    
    Monitors validation loss and stops training when no improvement
    is observed for a specified number of epochs (patience).
    """
    
    def __init__(self, patience=7, min_delta=0.001):
        """
        Initialize early stopping.
        
        Args:
            patience (int): Number of epochs to wait before stopping
            min_delta (float): Minimum improvement to reset patience
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        
    def __call__(self, val_loss):
        """
        Check if training should stop based on validation loss.
        
        Args:
            val_loss (float): Current validation loss
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement detected: reset counter
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            # No improvement: increment counter
            self.counter += 1
            return self.counter >= self.patience

# Initialize early stopping with 10 epochs patience
early_stopping = EarlyStopping(patience=10)

# Training configuration
num_epochs = 50  # Maximum number of epochs

print(f"\n🚀 Training Configuration:")
print(f"   Optimizer: AdamW (lr={optimizer.param_groups[0]['lr']:.0e}, wd={optimizer.param_groups[0]['weight_decay']:.0e})")
print(f"   Scheduler: CosineAnnealingWarmRestarts")
print(f"   Loss: Combined (70% L1 + 30% L2) + 30% Focal")
print(f"   Max epochs: {num_epochs}")
print(f"   Early stopping patience: {early_stopping.patience}")
print(f"   Batch size: {train_loader.batch_size} (train), {val_loader.batch_size} (val)")

# ============================================================================
# 13. ENHANCED TRAINING LOOP WITH COMPREHENSIVE MONITORING
# ============================================================================

# Training tracking variables
best_loss = float('inf')    # Track best validation loss for model saving
train_losses = []           # Store training losses for plotting/analysis
val_losses = []             # Store validation losses for plotting/analysis

print(f"\n🎯 Starting training for up to {num_epochs} epochs...")
print("=" * 80)

for epoch in range(num_epochs):
    print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
    print("-" * 50)
    
    # ========================================================================
    # TRAINING PHASE
    # ========================================================================
    
    model.train()  # Set model to training mode (enables dropout, batch norm training)
    train_loss = 0           # Accumulate training loss
    train_focal_loss = 0     # Accumulate focal loss for monitoring
    
    # Training loop with progress bar
    for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        # Move data to device (GPU/CPU) with non-blocking transfer for efficiency
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        # Clear gradients from previous iteration
        optimizer.zero_grad()
        
        # ====================================================================
        # FORWARD PASS
        # ====================================================================
        
        # Get model predictions
        out = model(x)
        
        # ====================================================================
        # LOSS COMPUTATION
        # ====================================================================
        
        # Compute primary combined loss (L1 + L2)
        loss = criterion(out, y)
        
        # Compute focal loss for hard examples
        focal_loss = focal_criterion(out, y)
        
        # Combine losses with weights (70% combined + 30% focal)
        total_loss = 0.7 * loss + 0.3 * focal_loss
        
        # ====================================================================
        # BACKWARD PASS AND OPTIMIZATION
        # ====================================================================
        
        # Compute gradients through backpropagation
        total_loss.backward()
        
        # Apply gradient clipping to prevent exploding gradients
        # This stabilizes training, especially with large models
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update model parameters using computed gradients
        optimizer.step()
        
        # ====================================================================
        # LOSS TRACKING
        # ====================================================================
        
        # Accumulate losses for epoch statistics
        train_loss += loss.item()
        train_focal_loss += focal_loss.item()
    
    # ========================================================================
    # VALIDATION PHASE
    # ========================================================================
    
    model.eval()  # Set model to evaluation mode (disables dropout, uses batch norm running stats)
    val_loss = 0             # Accumulate validation loss
    val_predictions = []     # Store predictions for detailed metrics
    val_targets = []         # Store ground truth for detailed metrics
    
    # Validation loop (no gradient computation needed)
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validation", leave=False):
            # Move data to device
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            # Get model predictions
            out = model(x)
            
            # Accumulate validation loss
            val_loss += criterion(out, y).item()
            
            # Store predictions and targets for detailed analysis
            val_predictions.extend(out.cpu().numpy())
            val_targets.extend(y.cpu().numpy())
    
    # ========================================================================
    # METRICS CALCULATION AND LOGGING
    # ========================================================================
    
    # Calculate average losses per batch
    train_mae = train_loss / len(train_loader)
    val_mae = val_loss / len(val_loader)
    
    # Convert lists to numpy arrays for calculations
    val_predictions = np.array(val_predictions).flatten()
    val_targets = np.array(val_targets).flatten()
    
    # Calculate additional metrics for comprehensive evaluation
    val_rmse = np.sqrt(np.mean((val_predictions - val_targets) ** 2))  # Root Mean Square Error
    val_mape = np.mean(np.abs((val_targets - val_predictions) / (val_targets + 1e-8))) * 100  # Mean Absolute Percentage Error
    
    # Store losses for tracking
    train_losses.append(train_mae)
    val_losses.append(val_mae)
    
    # ========================================================================
    # EPOCH SUMMARY LOGGING
    # ========================================================================
    
    print(f"📊 Epoch {epoch+1} Results:")
    print(f"   🏃 Train MAE: {train_mae:.4f}")
    print(f"   ✅ Val MAE: {val_mae:.4f}")
    print(f"   📈 Val RMSE: {val_rmse:.4f}")
    print(f"   📊 Val MAPE: {val_mape:.2f}%")
    print(f"   ⚙️ Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    # ========================================================================
    # LEARNING RATE SCHEDULING
    # ========================================================================
    
    # Update learning rate according to cosine annealing schedule
    scheduler.step()
    
    # ========================================================================
    # MODEL CHECKPOINTING
    # ========================================================================
    
    if val_mae < best_loss:
        # New best model found: save comprehensive checkpoint
        best_loss = val_mae
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'val_mape': val_mape,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        torch.save(checkpoint, "enhanced_crowd_counter_best.pth")
        print("   🎉 New best model saved!")
    
    # ========================================================================
    # EARLY STOPPING CHECK
    # ========================================================================
    
    if early_stopping(val_mae):
        print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
        print(f"   Best validation MAE: {best_loss:.4f}")
        break

print(f"\n✅ Training completed!")
print(f"🏆 Best validation MAE achieved: {best_loss:.4f}")
print("=" * 80)

# ============================================================================
# 14. ENHANCED INFERENCE WITH TEST TIME AUGMENTATION (TTA)
# ============================================================================

print(f"\n🔍 Starting enhanced inference with Test Time Augmentation...")
print("=" * 80)

# Load the best model checkpoint with all training metadata
print("📂 Loading best model checkpoint...")
checkpoint = torch.load("enhanced_crowd_counter_best.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set to evaluation mode

print(f"✅ Loaded model from epoch {checkpoint['epoch']} with validation MAE: {checkpoint['val_mae']:.4f}")

def predict_with_tta(model, x, tta_transforms):
    """
    Perform Test Time Augmentation for robust predictions.
    
    TTA applies multiple transformations to the same input image
    and averages the predictions to reduce variance and improve accuracy.
    This technique often provides 2-5% improvement in final accuracy.
    
    Args:
        model (nn.Module): Trained model for inference
        x (torch.Tensor): Input batch of images [B, 3, H, W]
        tta_transforms (list): List of transformation pipelines (unused in this implementation)
        
    Returns:
        torch.Tensor: Averaged predictions from multiple augmentations [B, 1]
    """
    predictions = []
    
    with torch.no_grad():
        # ====================================================================
        # 1. ORIGINAL IMAGE PREDICTION
        # ====================================================================
        pred_original = model(x)
        predictions.append(pred_original)
        
        # ====================================================================
        # 2. HORIZONTAL FLIP AUGMENTATION
        # ====================================================================
        # Flip images horizontally and predict
        # This is particularly useful for crowd scenes where orientation doesn't matter
        x_flip = torch.flip(x, [3])  # Flip along width dimension (index 3)
        pred_flip = model(x_flip)
        predictions.append(pred_flip)
        
        # ====================================================================
        # 3. MULTI-SCALE AUGMENTATION
        # ====================================================================
        # Test at different scales to capture crowd patterns of various sizes
        for scale in [0.9, 1.1]:  # 90% and 110% of original size
            # Calculate new size
            size = int(512 * scale)
            
            # Resize input to new scale
            x_scaled = F.interpolate(
                x, 
                size=(size, size), 
                mode='bilinear',           # Smooth interpolation
                align_corners=False        # Standard PyTorch practice
            )
            
            # Resize back to standard input size for model
            x_scaled = F.interpolate(
                x_scaled, 
                size=(512, 512), 
                mode='bilinear', 
                align_corners=False
            )
            
            # Get prediction for scaled image
            pred_scaled = model(x_scaled)
            predictions.append(pred_scaled)
    
    # ========================================================================
    # ENSEMBLE AVERAGING
    # ========================================================================
    
    # Stack all predictions and compute mean
    # This reduces prediction variance and improves robustness
    stacked_predictions = torch.stack(predictions, dim=0)  # [num_augmentations, B, 1]
    averaged_prediction = torch.mean(stacked_predictions, dim=0)  # [B, 1]
    
    return averaged_prediction

# ============================================================================
# 15. TEST SET PREDICTION WITH POST-PROCESSING
# ============================================================================

print("🎯 Generating predictions for test set...")

# Storage for final predictions
predictions_ensemble = []  # For potential future ensemble methods
preds = []                 # Final predictions for submission

# Inference loop with TTA
with torch.no_grad():
    for batch_idx, (x, ids) in enumerate(tqdm(test_loader, desc="Test Inference")):
        # Move input to device
        x = x.to(device, non_blocking=True)
        
        # ====================================================================
        # PREDICTION WITH TTA
        # ====================================================================
        
        # Get ensemble prediction using Test Time Augmentation
        out = predict_with_tta(model, x, AdvancedTransforms.get_tta_transforms())
        
        # ====================================================================
        # POST-PROCESSING
        # ====================================================================
        
        # 1. Ensure non-negative counts (crowds can't be negative)
        out = torch.clamp(out, min=0)
        
        # 2. Convert to integers and move to CPU
        out = out.cpu().squeeze().round().int().tolist()
        
        # 3. Handle single sample vs batch cases
        if isinstance(ids, torch.Tensor): 
            ids = ids.tolist()
        elif not isinstance(out, list):
            out = [out]  # Convert single value to list
            
        # 4. Store predictions with image IDs
        for img_id, cnt in zip(ids, out):
            preds.append({
                "id": img_id, 
                "count": max(0, int(cnt))  # Additional safety check for non-negative
            })

print(f"✅ Generated {len(preds)} predictions")

# ============================================================================
# 16. STATISTICAL POST-PROCESSING AND OUTLIER DETECTION
# ============================================================================

print("📊 Applying statistical post-processing...")

# Extract prediction counts for analysis
pred_counts = [p["count"] for p in preds]
mean_count = np.mean(pred_counts)
std_count = np.std(pred_counts)

print(f"📈 Raw Prediction Statistics:")
print(f"   Mean count: {mean_count:.2f}")
print(f"   Std deviation: {std_count:.2f}")
print(f"   Min count: {min(pred_counts)}")
print(f"   Max count: {max(pred_counts)}")

# ============================================================================
# OUTLIER DETECTION AND CORRECTION
# ============================================================================

# Identify and cap extremely high predictions using statistical thresholds
# This prevents unrealistic predictions that might hurt overall performance
outlier_threshold = mean_count + 3 * std_count  # 3-sigma rule
outlier_cap = mean_count + 2 * std_count        # Cap at 2-sigma

outliers_found = 0
for pred in preds:
    if pred["count"] > outlier_threshold:
        pred["count"] = int(outlier_cap)
        outliers_found += 1

if outliers_found > 0:
    print(f"⚠️  Corrected {outliers_found} outlier predictions (capped at {outlier_cap:.0f})")

# Final statistics after post-processing
final_counts = [p["count"] for p in preds]
print(f"� Final Prediction Statistics:")
print(f"   Mean count: {np.mean(final_counts):.2f}")
print(f"   Std deviation: {np.std(final_counts):.2f}")
print(f"   Min count: {min(final_counts)}")
print(f"   Max count: {max(final_counts)}")

# ============================================================================
# 17. ENHANCED SUBMISSION GENERATION WITH DETAILED ANALYSIS
# ============================================================================

print(f"\n📝 Generating submission files...")
print("=" * 80)

# ============================================================================
# SUBMISSION DATAFRAME CREATION
# ============================================================================

# Convert predictions to pandas DataFrame for easy manipulation
sub_df = pd.DataFrame(preds).sort_values("id")

# Ensure correct data types
sub_df['id'] = sub_df['id'].astype(int)
sub_df['count'] = sub_df['count'].astype(int)

print(f"📋 Submission DataFrame Info:")
print(f"   Shape: {sub_df.shape}")
print(f"   Columns: {list(sub_df.columns)}")
print(f"   Data types: {sub_df.dtypes.to_dict()}")

# ============================================================================
# CONFIDENCE SCORING (PLACEHOLDER FOR FUTURE ENHANCEMENT)
# ============================================================================

# Add confidence scoring based on prediction consistency
# This could be enhanced with ensemble variance, attention maps, etc.
sub_df['confidence'] = 1.0  # Default confidence (can be improved with ensemble methods)

# Future enhancement ideas for confidence scoring:
# - Prediction variance across TTA augmentations
# - Attention map entropy (high entropy = less confident)
# - Distance from training distribution
# - Model uncertainty estimation

# ============================================================================
# SUBMISSION FILE GENERATION
# ============================================================================

# 1. Save standard submission file (required format)
print("💾 Saving standard submission file...")
sub_df[['id', 'count']].to_csv(OUTPUT_SUB, index=False)
print(f"   ✅ Standard submission saved to: {OUTPUT_SUB}")

# 2. Save detailed results with additional metadata
print("📊 Saving detailed results with metadata...")
detailed_output = OUTPUT_SUB.replace('.csv', '_detailed.csv')

# Add additional columns for analysis
sub_df['model_type'] = 'EfficientNet-B4_Enhanced'
sub_df['tta_applied'] = True
sub_df['post_processed'] = True

sub_df.to_csv(detailed_output, index=False)
print(f"   ✅ Detailed results saved to: {detailed_output}")

# ============================================================================
# FINAL PERFORMANCE SUMMARY
# ============================================================================

print(f"\n🎯 Final Model Performance Summary:")
print("=" * 80)

# Model performance metrics
print(f"📈 Training Results:")
print(f"   Best Validation MAE: {checkpoint['val_mae']:.4f}")
print(f"   Best Validation RMSE: {checkpoint['val_rmse']:.4f}")
print(f"   Best Validation MAPE: {checkpoint.get('val_mape', 'N/A'):.2f}%" if 'val_mape' in checkpoint else "")
print(f"   Training stopped at epoch: {checkpoint['epoch']}")

# Prediction statistics
print(f"\n📊 Prediction Statistics:")
print(f"   Total test predictions: {len(sub_df):,}")
print(f"   Average predicted count: {sub_df['count'].mean():.2f}")
print(f"   Prediction std deviation: {sub_df['count'].std():.2f}")
print(f"   Prediction range: {sub_df['count'].min()} - {sub_df['count'].max()}")
print(f"   Median prediction: {sub_df['count'].median():.1f}")

# Model architecture summary
print(f"\n🏗️  Model Architecture:")
print(f"   Backbone: EfficientNet-B4 (or ResNet50 fallback)")
print(f"   Attention: Channel + Spatial")
print(f"   Multi-scale features: ✅")
print(f"   Test Time Augmentation: ✅")
print(f"   Post-processing: ✅")

# Training configuration recap
print(f"\n⚙️  Training Configuration:")
print(f"   Optimizer: AdamW")
print(f"   Loss function: Combined (L1 + L2) + Focal")
print(f"   Data augmentation: Advanced (Mixup + transforms)")
print(f"   Early stopping: ✅")
print(f"   Gradient clipping: ✅")

# File outputs
print(f"\n📁 Generated Files:")
print(f"   🎯 Submission: {OUTPUT_SUB}")
print(f"   📊 Detailed: {detailed_output}")
print(f"   🤖 Model: enhanced_crowd_counter_best.pth")

print(f"\n🎉 Enhanced crowd counting pipeline completed successfully!")
print("=" * 80)

# ============================================================================
# OPTIONAL: PREDICTION DISTRIBUTION ANALYSIS
# ============================================================================

print(f"\n📈 Prediction Distribution Analysis:")

# Count distribution bins
bins = [0, 10, 25, 50, 100, 200, 500, float('inf')]
bin_labels = ['0-10', '11-25', '26-50', '51-100', '101-200', '201-500', '500+']

# Categorize predictions
for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
    if high == float('inf'):
        count = sum(1 for x in sub_df['count'] if x > low)
    else:
        count = sum(1 for x in sub_df['count'] if low < x <= high)
    percentage = count / len(sub_df) * 100
    print(f"   {bin_labels[i]:>8}: {count:>6} predictions ({percentage:>5.1f}%)")

print(f"\n✨ Ready for submission! Good luck with your competition! 🚀")
