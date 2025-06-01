import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class CrowdDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir

        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor()
        ])

        self.data['filename'] = self.data['id'].apply(lambda x: f"seq_{int(x):06d}.jpg")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])

        image = Image.open(img_path).convert("RGB")
        count = torch.tensor(row['count'], dtype=torch.float32)  # 👈 Fix here

        image = self.transform(image)

        return image, count
