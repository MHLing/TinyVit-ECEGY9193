import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.tiny_vit import tiny_vit_5m_224  # TinyViT-5M 构造函数

def get_dataloaders(data_root, batch_size=64):
    # CIFAR-100 是 32x32，这里统一 resize 到 224x224 给 TinyViT 用
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],  # CIFAR-100 经验均值
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

    train_set = datasets.CIFAR100(
        root=data_root,
        train=True,
        download=True,          # 没有数据集会自动下载
        transform=transform_train,
    )

    test_set = datasets.CIFAR100(
        root=data_root,
        train=False,
        download=True,
        transform=transform_test,
    )

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
    data_root = r"C:\Users\lmh98\Desktop\CV Project\data\cifar-100-python"   # 数据会下在 TinyViT/cifar100_data 下面
    batch_size = 64
    epochs = 50                     # 先跑 50 个 epoch 看看情况
    lr = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ===== 数据加载器 =====
    train_loader, test_loader = get_dataloaders(data_root, batch_size=batch_size)

    # ===== 构建 TinyViT-5M 模型 =====
    # num_classes=100 对应 CIFAR-100
    model = tiny_vit_5m_224(pretrained=True, num_classes=100)
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
            torch.save(model.state_dict(), "tinyvit5m_cifar100_best.pth")
            print(f"*** New best acc: {best_acc:.2f}%, checkpoint saved.")

    print("Training finished. Best Acc: {:.2f}%".format(best_acc))


if __name__ == "__main__":
    main()
