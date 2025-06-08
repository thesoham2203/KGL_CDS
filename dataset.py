import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np

class CrowdDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, log_transform=True):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.log_transform = log_transform

        # Normalize using ImageNet statistics + basic augmentation for training
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Optional augmentation
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.data['filename'] = self.data['id'].apply(
            lambda x: f"seq_{int(x):06d}.jpg"
        )

        self.ids = self.data['filename'].tolist()  # for external reference if needed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # 🔁 Log-transform the count for better regression stability
        count = row['count']
        if self.log_transform:
            count = np.log1p(count)

        count = torch.tensor(count, dtype=torch.float32)

        return image, count
