from pathlib import Path
import argparse, json
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import SkinCNN

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='Train the educational skin-screening CNN.')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--image-size', type=int, default=128)
    args = parser.parse_args()

    train_dir = ROOT / 'dataset' / 'train'
    if not train_dir.exists():
        raise SystemExit(f'Dataset folder not found: {train_dir}')
    tfm = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    full = datasets.ImageFolder(train_dir, transform=tfm)
    if len(full.classes) < 2 or len(full) < 4:
        raise SystemExit('Add at least 2 class folders and 2+ images per class under dataset/train/.')
    val_len = max(1, int(0.2 * len(full)))
    train_len = len(full) - val_len
    train_set, val_set = random_split(full, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SkinCNN(len(full.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_acc = -1.0
    out = ROOT / 'models' / 'skin_cnn.pth'
    out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train(); running = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(); loss = criterion(model(images), labels); loss.backward(); optimizer.step()
            running += loss.item() * images.size(0)
        model.eval(); correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                pred = model(images).argmax(1); correct += (pred == labels).sum().item(); total += labels.size(0)
        val_acc = correct / max(total, 1)
        print(f'Epoch {epoch+1}/{args.epochs} | train_loss={running/max(train_len,1):.4f} | val_accuracy={val_acc:.3f}')
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'state_dict': model.state_dict(), 'classes': full.classes,
                        'image_size': args.image_size, 'validation_accuracy': best_acc}, out)
    print(f'Checkpoint saved: {out}')
    print('NOTE: This is an educational baseline. Validate on independent, representative clinical data before any real-world use.')

if __name__ == '__main__': main()
