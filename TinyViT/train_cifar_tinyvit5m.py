import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from datasets import load_dataset

from models.tiny_vit import tiny_vit_5m_224, ViTSmallBaseline  # TinyViT-5M 构造函数

class HFCIFAR100(Dataset):
    """Wrap HuggingFace CIFAR-100 dataset to work with PyTorch + torchvision transforms."""

    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]
        image = example["img"]          # PIL.Image
        label = int(example["fine_label"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_dataloaders(data_name, batch_size=64):
    # CIFAR-100 original size is 32x32, resize to 224x224 for TinyViT
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],  # CIFAR-100 mean
            std=[0.2675, 0.2565, 0.2761],
        ),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761],
        ),
    ])

    # Load CIFAR-100 from HuggingFace
    hf_dataset = load_dataset(data_name)
    hf_train = hf_dataset["train"]
    hf_test = hf_dataset["test"]

    train_set = HFCIFAR100(hf_train, transform=transform_train)
    test_set = HFCIFAR100(hf_test, transform=transform_test)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch}] Step [{i+1}/{len(loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    epoch_loss = running_loss / total
    epoch_acc = correct / total * 100.0
    print(f"[Train] Epoch {epoch}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%")
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total * 100.0
    print(f"[Eval ] Epoch {epoch}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%")
    return epoch_loss, epoch_acc


def main():
    # ===== 基本配置 =====
    batch_size = 128
    epochs = 50                     # 先跑 50 个 epoch 看看情况
    lr = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ===== 数据加载器 =====
    train_loader, test_loader = get_dataloaders("uoft-cs/cifar100", batch_size=batch_size)

    # ===== 构建 TinyViT-5M 模型 =====
    # num_classes=100 对应 CIFAR-100
    model = ViTSmallBaseline()
    # model = tiny_vit_5m_224(pretrained=True, num_classes=100)
    model.to(device)

    # ===== 损失函数 & 优化器 =====
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        _, val_acc = evaluate(model, test_loader, criterion, device, epoch)

        # 简单保存最好 ckpt
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "smallvit5m_cifar100_best.pth")
            print(f"*** New best acc: {best_acc:.2f}%, checkpoint saved.")

    print("Training finished. Best Acc: {:.2f}%".format(best_acc))


if __name__ == "__main__":
    main()
