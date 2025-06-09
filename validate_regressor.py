import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from dataset import CrowdDataset

# Model Definition
class CountRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet18(weights=None)
        self.base.fc = nn.Linear(self.base.fc.in_features, 1)

    def forward(self, x):
        return self.base(x)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform
val_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Corrected dataset init
val_dataset = CrowdDataset("val_labels.csv", "val", transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Load Model
model = CountRegressor().to(device)
model.load_state_dict(torch.load("regressor_best_model.pth", map_location=device))
model.eval()

# Evaluate
mae = 0.0
with torch.no_grad():
    for imgs, targets in val_loader:
        imgs, targets = imgs.to(device), targets.to(device)
        outputs = model(imgs).view(-1)
        mae += torch.abs(outputs - targets).sum().item()

mae /= len(val_dataset)
print(f"📊 Validation MAE (Mean Absolute Error): {mae:.2f} people")
