import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from model import CSRNet
from dataset import CrowdDataset
import random

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform for validation images
val_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load validation dataset
val_dataset = CrowdDataset("val_labels.csv", "val", transform=val_transform)

# Load model
model = CSRNet(load_pretrained=False).to(device)
model.load_state_dict(torch.load("best_model_newest.pth", map_location=device))
model.eval()

# Denormalization for visualization
def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return (img_tensor * std + mean).clamp(0, 1)

# Evaluate on 5 random validation images
indices = random.sample(range(len(val_dataset)), 5)

for idx in indices:
    img, actual_count = val_dataset[idx]
    img_tensor = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        predicted_count = output.sum().item()

    # Convert tensor to NumPy image
    img_np = denormalize(img).permute(1, 2, 0).cpu().numpy()

    # Plot image with counts
    plt.figure(figsize=(5, 5))
    plt.imshow(img_np)
    plt.title(f"Predicted: {predicted_count:.2f} | Actual: {actual_count}")
    plt.axis("off")
    plt.show()
