import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from model import CSRNet
from dataset import CrowdDataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Data augmentation (train) + normalization
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((320, 320), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 2) Validation transform: just resize + normalize
    val_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 3) Load datasets
    train_dataset = CrowdDataset("training_data.csv", "train", transform=train_transform)
    val_dataset   = CrowdDataset("val_labels.csv", "val",    transform=val_transform)

    # 4) DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=12, shuffle=False,
                              num_workers=4, pin_memory=True)

    # 5) Model, loss, optimizer, scheduler
    model = CSRNet(load_pretrained=True).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(1, 61):
        model.train()
        train_loss = 0.0

        for imgs, counts in train_loader:
            imgs   = imgs.to(device)
            counts = counts.to(device)

            preds_map = model(imgs)
            preds_cnt = preds_map.sum(dim=(2, 3))

            loss = criterion(preds_cnt.view(-1), counts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)

        # Validation pass
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, counts in val_loader:
                imgs   = imgs.to(device)
                counts = counts.to(device)

                preds_map = model(imgs)
                preds_cnt = preds_map.sum(dim=(2, 3))

                loss = criterion(preds_cnt.view(-1), counts)
                val_loss += loss.item() * imgs.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, "
              f"Val Loss = {avg_val_loss:.4f}")

        # Early stopping + checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), "best_model_newest.pth")
            print("✅ Model saved!")
        else:
            early_stop_counter += 1
            if early_stop_counter > 7:
                print("⏹️ Early stopping triggered.")
                break

    print("🎉 Training complete.")


if __name__ == "__main__":
    main()
