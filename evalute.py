import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from model import CSRNet
from dataset import CrowdDataset
from PIL import Image
import random

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CSRNet().to(device)
model.load_state_dict(torch.load("best_model_new.pth", map_location=device))
model.eval()

# Define transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load validation dataset
val_dataset = CrowdDataset("val_labels.csv", "val", transform=transform)

# Choose N random images to visualize
N = 5
indices = random.sample(range(len(val_dataset)), N)

for idx in indices:
    img_tensor, true_count = val_dataset[idx]
    img = transforms.ToPILImage()(img_tensor.cpu())

    # Move to batch and device
    with torch.no_grad():
        pred = model(img_tensor.unsqueeze(0).to(device))
        pred_count = pred.sum().item()

    # Show result
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(f"Predicted: {pred_count:.1f}, Actual: {true_count:.1f}")
    plt.axis('off')
    plt.show()
