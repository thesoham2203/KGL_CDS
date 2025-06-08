import torch
from torchvision import transforms
import pandas as pd
from PIL import Image
from model import CSRNet
import os
from tqdm import tqdm

# Configs
val_csv = "val_labels.csv"
val_dir = "val"
model_path = "best_model_newest.pth"
output_csv = "results.csv"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = CSRNet(load_pretrained=False).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Load CSV
df = pd.read_csv(val_csv)

# Transform
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Collect results
results = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    img_id = int(row['id'])
    actual = float(row['count'])

    # Format: seq_001501.jpg
    filename = f"seq_{img_id:06d}.jpg"
    img_path = os.path.join(val_dir, filename)

    if not os.path.exists(img_path):
        print(f"⚠️ Image not found: {img_path}")
        continue

    # Load + preprocess
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_map = model(image)
        predicted = pred_map.sum().item()

    acc = max(0, 100 - abs(predicted - actual) / (actual + 1e-6) * 100)
    results.append({
        "image_id": filename,
        "actual_count": round(actual, 2),
        "predicted_count": round(predicted, 2),
        "accuracy(%)": round(acc, 2)
    })

# Save CSV
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv, index=False)
print(f"\n✅ Results saved to: {output_csv}")
