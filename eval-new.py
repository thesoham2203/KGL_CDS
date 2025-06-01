import torch
from model import CSRNet
from dataset import CrowdDataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CSRNet()
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device).eval()

val_dataset = CrowdDataset("val_images", "val_labels.xlsx")
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

mae = 0.0
mse = 0.0

with torch.no_grad():
    for imgs, counts in val_loader:
        imgs, counts = imgs.to(device), counts.to(device)
        preds = model(imgs).sum(dim=[2, 3]).view(-1)
        mae += torch.abs(preds - counts).sum().item()
        mse += ((preds - counts) ** 2).sum().item()

n = len(val_dataset)
print(f"MAE: {mae/n:.2f}, RMSE: {(mse/n) ** 0.5:.2f}")
