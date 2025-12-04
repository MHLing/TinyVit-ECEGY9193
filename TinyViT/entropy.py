import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from datasets import load_dataset
from models.tiny_vit import tiny_vit_5m_224, TinyViTBlock  # TinyViT-5M 构造函数

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


def get_dataloaders(
    data_name,
    batch_size=64,
    num_train_samples=None,
    num_test_samples=None,
    seed: int = 42,
):
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

    # Load dataset from HuggingFace
    hf_dataset = load_dataset(data_name)
    hf_train = hf_dataset["train"]
    hf_test = hf_dataset["test"]

    train_set = HFCIFAR100(hf_train, transform=transform_train)
    test_set = HFCIFAR100(hf_test, transform=transform_test)

    # ---- optional: subsample train/test by size ----
    g = torch.Generator().manual_seed(seed)

    if num_train_samples is not None:
        n_train = min(num_train_samples, len(train_set))
        idx_train = torch.randperm(len(train_set), generator=g)[:n_train].tolist()
        train_set = Subset(train_set, idx_train)

    if num_test_samples is not None:
        n_test = min(num_test_samples, len(test_set))
        idx_test = torch.randperm(len(test_set), generator=g)[:n_test].tolist()
        test_set = Subset(test_set, idx_test)

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

# ---------- 1. Entropy helper ----------
def compute_entropy_from_logits_tensor(logits: torch.Tensor, dim: int = -1) -> float:
    """
    Compute mean entropy from logits along the given dimension.
    logits: Tensor of shape [..., num_classes]
    dim:    dimension corresponding to class/logit axis.
    """
    logits = logits.float()
    probs = torch.softmax(logits, dim=dim)
    probs = probs.clamp_min(1e-12)
    log_probs = torch.log(probs)
    entropy = -(probs * log_probs).sum(dim=dim)  # shape: [... except dim]
    return entropy.mean().item()


def compute_entropy_from_list(logits_list, dim: int = -1) -> float:
    """
    logits_list: list of Tensors collected from one block across the testloader.
                 We concatenate along batch dimension.
    """
    if len(logits_list) == 0:
        return float("nan")

    # concatenate along batch dimension
    all_logits = torch.cat(logits_list, dim=0)  # assume dim 0 is batch
    return compute_entropy_from_logits_tensor(all_logits, dim=dim)


# ---------- 2. Register hooks to collect logits ----------
def register_tinyvitblock_collect_logits(model: nn.Module):
    """
    Register forward hooks on all TinyViTBlock modules.
    For each block, we create a list to store its logits across the whole testloader.

    Returns:
        block_logits: dict[name] -> list of logits tensors
        hooks:        list of hook handles (for later removal)
    """
    block_logits = {}
    hooks = []

    def make_hook(name):
        # create a list for this block
        block_logits[name] = []

        def hook(module, input, output):
            logits = output
            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            # detach graph, move to cpu to save GPU memory
            logits = logits.detach().cpu()
            block_logits[name].append(logits)

        return hook

    for name, module in model.named_modules():
        if isinstance(module, TinyViTBlock):
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    print(f"Registered {len(hooks)} TinyViTBlock hooks (collect logits).")
    return block_logits, hooks


# ---------- 3. Compute entropy for all blocks ----------
def compute_all_block_entropies(block_logits, dim: int = -1):
    """
    block_logits: dict[name] -> list of logits tensors
    dim:          class/logit dimension (usually -1)
    Returns:
        entropies: dict[name] -> entropy (float)
    """
    entropies = {}
    for name, logs in block_logits.items():
        ent = compute_entropy_from_list(logs, dim=dim)
        entropies[name] = ent
    return entropies


# ---------- 4. Example usage with your testloader ----------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) build your model
    # from models.tiny_vit import tiny_vit_5m_224
    ckpt_path = "tinyvit5m_cifar100_best.pth"
    dataset_name = "uoft-cs/cifar100"
    batch_size = 64
    num_test_samples = 1000
    model = tiny_vit_5m_224(pretrained=True, num_classes=100)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    model.eval()

    # 2) register hooks
    block_logits, hooks = register_tinyvitblock_collect_logits(model)

    # 3) you already have a testloader with 50 batches
    _, test_loader = get_dataloaders(dataset_name, batch_size=batch_size, num_test_samples=num_test_samples)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            _ = model(images)

    # 4) remove hooks (optional but recommended)
    for h in hooks:
        h.remove()

    # 5) after running through the whole testloader, compute entropy for each block
    entropies = compute_all_block_entropies(block_logits, dim=-1)

    print("\n=== TinyViTBlock entropies (from all collected logits) ===")
    for name, ent in entropies.items():
        print(f"{name}: {ent:.4f}")
