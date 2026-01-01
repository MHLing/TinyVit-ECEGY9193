import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from datasets import load_dataset
from models.tiny_vit import tiny_vit_5m_224, TinyViTBlock, tiny_vit_small  # TinyViT-5M 构造函数
import torch.nn.functional as F
from tqdm.auto import tqdm


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

def register_forward_hooks(model):
    outputs = {}   # dict: name -> tensor

    def get_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                outputs[name] = output.detach()
            elif isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                outputs[name] = tuple(o.detach() for o in output)
            return
        return hook

    for name, module in model.named_modules():
        if name == "":
            continue
        module.register_forward_hook(get_hook(name))

    return outputs

def kl_loss_fn(student_logits, teacher_logits, T=1.0):
    teacher_probs = F.softmax(teacher_logits / T, dim=-1)
    student_logprobs = F.log_softmax(student_logits / T, dim=-1)
    loss = F.kl_div(student_logprobs, teacher_probs, reduction="batchmean") * (T * T)
    return loss

def layer_distill_loss(student_outputs, teacher_outputs, layer_map, weight=1.0):
    loss = 0.0
    count = 0

    for s_key, t_key in layer_map.items():

        if s_key not in student_outputs:
            continue
        if t_key not in teacher_outputs:
            continue

        s_out = student_outputs[s_key]
        t_out = teacher_outputs[t_key]

        # 如果输出是 tuple（例如 multi-head attn 的多个东西）
        if isinstance(s_out, tuple):
            s_out = s_out[0]
        if isinstance(t_out, tuple):
            t_out = t_out[0]

        # shape mismatch: skip
        if s_out.shape != t_out.shape:
            # print(f"[Skip] Shape mismatch for {s_key} vs {t_key}: {s_out.shape} != {t_out.shape}")
            continue

        loss += F.mse_loss(s_out, t_out)
        count += 1

    if count > 0:
        loss = loss / count

    return loss * weight


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = "/scratch/xl5444/workspace/TinyVit-ECEGY9193/TinyViT/finetune/tinyvit5m_imagenet100_best.pth"
    dataset_name = "clane9/imagenet-100"
    batch_size = 256
    num_test_samples = 1000
    epochs = 10
    token_pruning_ratio = 0.2  # 剪枝 20% 的 tokens (可调整: 0.0-0.5)
    token_pruning_method = 'magnitude'  # 'attention' 或 'magnitude'

    layer_map = {
        "layers.0.blocks.0": "layers.0.blocks.0",
        "layers.0.blocks.1": "layers.0.blocks.1",
        "layers.1.blocks.0": "layers.1.blocks.0",
        "layers.1.blocks.1": "layers.1.blocks.1",
        "layers.2.blocks.0": "layers.2.blocks.2",
        "layers.2.blocks.1": "layers.2.blocks.4",
        "layers.3.blocks.0": "layers.3.blocks.0",
        "layers.3.blocks.1": "layers.3.blocks.1",
    }

    # teacher & student
    model_baseline = tiny_vit_5m_224(pretrained=True, num_classes=100).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model_baseline.load_state_dict(state_dict, strict=True)

    model = tiny_vit_small(pretrained=True, num_classes=100, token_pruning_ratio=token_pruning_ratio, token_pruning_method=token_pruning_method).to(device)
    model.load_state_dict(state_dict, strict=False)

    model_baseline.eval()
    model.train()

    # --- 注册 HOOK：每个 hook 会自动把输出保存进 dict ---
    teacher_outputs = register_forward_hooks(model_baseline)
    student_outputs = register_forward_hooks(model)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    train_loader, test_loader = get_dataloaders(
        dataset_name, batch_size=batch_size
    )

    T = 2.0
    alpha = 0.8
    alpha_fea = 0.2

    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

        for images, labels in pbar:

            images = images.to(device)
            labels = labels.to(device)

            # -----  每个 batch 开始时清空 dict  -----
            teacher_outputs.clear()
            student_outputs.clear()

            # --- teacher logits ---
            with torch.no_grad():
                teacher_logits = model_baseline(images)

            # --- student logits ---
            student_logits = model(images)

            # --- losses ---
            ce_loss = F.cross_entropy(student_logits, labels)
            kd_loss = kl_loss_fn(student_logits, teacher_logits, T=T)

            feat_loss = layer_distill_loss(student_outputs, teacher_outputs, layer_map, weight=1.0)
            loss = alpha * kd_loss + (1 - alpha) * ce_loss + alpha_fea * feat_loss
            # loss = alpha * kd_loss + (1 - alpha) * ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                f"CE={ce_loss.item():.4f}, "
                f"KL={kd_loss.item():.4f}, "
                f"FEAT={feat_loss.item():.4f}, "
                f"Total={loss.item():.4f}"
            )


    torch.save(model.state_dict(), "student_model_imagenet100_random.pth")

    return student_outputs


if __name__ == "__main__":
    train()



    