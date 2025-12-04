import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from datasets import load_dataset
import time

from models.tiny_vit import tiny_vit_5m_224, tiny_vit_small  # TinyViT-5M 构造函数

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
    model = tiny_vit_5m_224(pretrained=True, num_classes=100)
    model.to(device)
    images_test = []

    # warmup
    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            if i >= 10 :
                continue
            images = images.to(device, non_blocking=True)
            _ = model(images)
            images_test = images

    start = time.time()
    outputs = model(images_test)
    end = time.time()
    print("Single forward latency: {:.6f} seconds".format(end - start))
    print("Latency per image: {:.6f} seconds".format((end - start)/images.size(0)))
    
    # with torch.no_grad():
    #     for i, (images, targets) in enumerate(test_loader):
    #         if i >= 1:
    #             continue
    #         images = images.to(device, non_blocking=True)
    #         _ = model(images)

    # print("Block times (seconds):")
    # for i, t in enumerate(model.block_times):
    #     print(f"Block {i}: {t} ms")

    # _, val_acc = evaluate(model, test_loader, criterion, device, epoch)


def eval(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128

    model = tiny_vit_small(pretrained=True, num_classes=100).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()
    _, test_loader = get_dataloaders("uoft-cs/cifar100", batch_size=batch_size)

    correct = 0
    total = 0
    images_test = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    start = time.time()
    outputs = model(images)
    end = time.time()
    print("Single forward latency: {:.6f} seconds".format(end - start))
    print("Latency per image: {:.6f} seconds".format((end - start)/images.size(0)))

    epoch_acc = correct / total * 100.0
    print(f"[Eval ] Acc={epoch_acc:.2f}%")
    return epoch_acc

if __name__ == "__main__":
    main()
    # val_acc = eval("/scratch/xl5444/workspace/TinyVit-ECEGY9193/TinyViT/finetune/student_model.pth")
