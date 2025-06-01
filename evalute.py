import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset class
class ValCrowdDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_id = int(row['image_id'])  # ensure it's an integer
        img_filename = f"seq_{image_id:06d}.jpg"  # format: seq_000001.jpg
        img_path = os.path.join(self.img_dir, img_filename)  # this is now a valid string

        image = Image.open(img_path).convert("RGB")
        count = torch.tensor([row['count']], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, count, img_filename  # return string as image_id


# Model definition
class ResNetRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.backbone(x)

# Load validation labels
df_val = pd.read_csv("val_labels.csv")
df_val.columns = ['image_id', 'count']

# Transforms
val_transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Dataset & Loader
val_dataset = ValCrowdDataset(df_val, "val", val_transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Load model
model = ResNetRegressor().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Inference
results = []
with torch.no_grad():
    for images, counts, image_ids in tqdm(val_loader, desc="Evaluating"):
        images = images.to(device)
        preds = model(images).cpu().squeeze().numpy()
        actual = counts.squeeze().numpy()

        # Ensure numpy arrays
        preds = np.atleast_1d(preds)
        actual = np.atleast_1d(actual)

        for img_id, gt, pred in zip(image_ids, actual, preds):
            acc = 100 - (abs(gt - pred) / (gt + 1e-5)) * 100  # prevent division by zero
            results.append({
                "image_id": img_id,
                "actual_count": round(gt, 2),
                "predicted_count": round(pred, 2),
                "accuracy(%)": round(acc, 2)
            })

# Save results
out_df = pd.DataFrame(results)
out_df.to_csv("final_output.csv", index=False)
print("\n✅ Output saved as final_output.csv")
