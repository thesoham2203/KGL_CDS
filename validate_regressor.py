import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
import pandas as pd
from dataset import CrowdDataset
import os
from tqdm import tqdm

class CountRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet18(weights='IMAGENET1K_V1')
        self.base.fc = nn.Linear(self.base.fc.in_features, 1)

    def forward(self, x):
        return self.base(x)

# ----- Load data -----
val_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_dataset = CrowdDataset("val_labels.csv", "val", transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# ----- Load model -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CountRegressor().to(device)
model.load_state_dict(torch.load("regressor_best_model.pth", map_location=device))
model.eval()

# ----- Predict & Write -----
results = []
total_accuracy = 0.0

with torch.no_grad():
    for idx, (img, actual) in enumerate(tqdm(val_loader)):
        img = img.to(device)
        actual_count = actual.item()

        predicted = model(img).item()
        predicted_count = max(0, round(predicted))  # avoid negatives
        accuracy = 100 - (abs(predicted_count - actual_count) / actual_count * 100 if actual_count != 0 else 0)
        total_accuracy += accuracy

        # Convert idx back to image name: 1501 → seq_001501.jpg
        image_id = val_dataset.ids[idx]
        filename = image_id  # Already in 'seq_00XXXX.jpg' format


        results.append({
            "image_id": filename,
            "actual_count": actual_count,
            "predicted_count": predicted_count,
            "accuracy(%)": round(accuracy, 2)
        })

# ----- Save results -----
df = pd.DataFrame(results)
df.to_csv("regressor_results.csv", index=False)

avg_accuracy = total_accuracy / len(val_dataset)
print(f"\n✅ Results saved to regressor_results.csv")
print(f"📊 Average Accuracy: {avg_accuracy:.2f}%")
