import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from datasets import load_dataset
import time
import torch.nn as nn

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


class HFImageNet(Dataset):
    """Wrap HuggingFace ImageNet dataset."""
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]
        image = example["image"]         # PIL.Image
        label = int(example["label"])

        if self.transform is not None:
            image = self.transform(image)
        return image, label


def get_dataloaders(data_name, batch_size=64, size=50000):

    # CIFAR-100 normalization (default branch)
    cifar_mean = [0.5071, 0.4867, 0.4408]
    cifar_std = [0.2675, 0.2565, 0.2761]

    # ImageNet normalization
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Shared transforms
    transform_train_cifar = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar_mean, std=cifar_std),
    ])

    transform_test_cifar = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar_mean, std=cifar_std),
    ])

    transform_train_imagenet = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    transform_test_imagenet = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    
    if "imagenet" in data_name.lower():
        hf_dataset = load_dataset(data_name)
        hf_train = hf_dataset["train"]
        hf_test = hf_dataset["validation"]

        if size is not None:
            hf_train = hf_train.shuffle(seed=42).select(range(size))

        train_set = HFImageNet(hf_train, transform=transform_train_imagenet)
        test_set = HFImageNet(hf_test, transform=transform_test_imagenet)

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

        return train_loader, test_loader

    # CIFAR-100 default branch
    hf_dataset = load_dataset(data_name)
    hf_train = hf_dataset["train"]
    hf_test = hf_dataset["test"]

    train_set = HFCIFAR100(hf_train, transform=transform_train_cifar)
    test_set = HFCIFAR100(hf_test, transform=transform_test_cifar)

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


def evaluate(model, loader, criterion, device):
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
    print(f"[Eval ]: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%")
    return epoch_loss, epoch_acc


def baseline(batch_size, ckpt_path, dataset_name):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders(dataset_name, batch_size=batch_size)

    # ===== 构建 TinyViT-5M 模型 =====
    # num_classes=100 对应 CIFAR-100
    model = tiny_vit_5m_224(pretrained=True, num_classes=100)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
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
    print("Single forward latency: {:.6f} seconds".format((end - start)*1000))
    print("Latency per image: {:.6f} seconds".format((end - start)*1000/images.size(0)))
    
    # with torch.no_grad():
    #     for i, (images, targets) in enumerate(test_loader):
    #         if i >= 1:
    #             continue
    #         images = images.to(device, non_blocking=True)
    #         _ = model(images)

    # print("Block times (seconds):")
    # for i, t in enumerate(model.block_times):
    #     print(f"Block {i}: {t} ms")
    criterion = nn.CrossEntropyLoss()
    _, val_acc = evaluate(model, test_loader, criterion, device)


def evalutation(dataset_name, batch_size, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    token_pruning_ratio = 0.0  # 剪枝 20% 的 tokens (可调整: 0.0-0.5)
    token_pruning_method = 'magnitude'  # 'attention' 或 'magnitude'
    model = tiny_vit_small(pretrained=True, num_classes=100, token_pruning_ratio=token_pruning_ratio, token_pruning_method=token_pruning_method).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()
    _, test_loader = get_dataloaders(dataset_name, batch_size=batch_size)

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
    print("Single forward latency: {:.6f} seconds".format((end - start)*1000))
    print("Latency per image: {:.6f} seconds".format((end - start)*1000/images.size(0)))

    epoch_acc = correct / total * 100.0
    print(f"[Eval ] Acc={epoch_acc:.2f}%")
    return epoch_acc

if __name__ == "__main__":
    batch_size = 256
    baseline_path = "/scratch/xl5444/workspace/TinyVit-ECEGY9193/TinyViT/tinyvit5m_cifar100_best.pth"
    ckpt_path = "/scratch/xl5444/workspace/TinyVit-ECEGY9193/TinyViT/finetune/student_model_cifar100_random.pth"
    dataset_name = "uoft-cs/cifar100" # uoft-cs/cifar100 clane9/imagenet-100

    print("-"*20)
    print("baseline:")
    baseline(dataset_name=dataset_name, batch_size=batch_size, ckpt_path=baseline_path)
    print("-"*20)
    print("our:")
    evalutation(dataset_name=dataset_name, batch_size=batch_size, model_path=ckpt_path)
