import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from Magnitude_based_VitSmall import MagnitudePruneViTSmall
from Attention_based_VitSmall import AttentionPruneViTSmall
from Baseline_VitSmall import ViTSmallBaseline


# ---------------- CIFAR-100 Loader ----------------
def get_cifar100(batch_size=64):
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        ),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        ),
    ])

    train_set = datasets.CIFAR100("./data", train=True, download=True, transform=transform_train)
    test_set  = datasets.CIFAR100("./data", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader


# ---------------- Train One Epoch (with ETA) ----------------
def train_one_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs, global_start_time):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    epoch_start_time = time.time()
    num_iters = len(loader)

    for i, (imgs, labels) in enumerate(loader):
        iter_start = time.time()

        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Accuracy
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
        total_loss += loss.item()

        # Iter time
        iter_time = time.time() - iter_start

        # ETA (this epoch)
        iters_done = i + 1
        iters_left = num_iters - iters_done
        eta_iters = iters_left * iter_time

        # Global ETA
        progress = (epoch + iters_done / num_iters) / total_epochs
        elapsed = time.time() - global_start_time

        if progress > 0:
            eta_global = elapsed * (1 / progress - 1)
        else:
            eta_global = 0

        if i % 50 == 0:
            print(
                f"[Epoch {epoch+1}] Iter {i}/{num_iters} "
                f"Loss={loss.item():.4f} IterTime={iter_time:.3f}s "
                f"ETA(iter)={eta_iters/60:.1f}m ETA(total)={eta_global/3600:.2f}h"
            )

    epoch_time = time.time() - epoch_start_time
    train_acc = 100.0 * correct / total
    return total_loss / len(loader), train_acc, epoch_time



# ---------------- Test ----------------
def test(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    return total_loss / len(loader), acc


# ---------------- Main ----------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    global_start_time = time.time()

    # Choose model:
    # model = ViTSmallBaseline()
    model = MagnitudePruneViTSmall()
    # model = AttentionPruneViTSmall()

    model = model.to(device)
    train_loader, test_loader = get_cifar100(batch_size=32)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    EPOCHS = 10
    total_train_time = 0
    best_acc = 0.0   
    for epoch in range(EPOCHS):
        print(f"\n========== Epoch {epoch+1}/{EPOCHS} ==========")

        train_loss, train_acc, epoch_time = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, EPOCHS, global_start_time
        )
        total_train_time += epoch_time

        test_loss, test_acc = test(model, test_loader, criterion, device)

        print(f"Epoch Time: {epoch_time:.2f} sec ({epoch_time/60:.2f} min)")
        print(f" Train Loss: {train_loss:.4f},  Train Acc: {train_acc:.2f}%")
        print(f" Test  Loss: {test_loss:.4f},  Test  Acc: {test_acc:.2f}%")

        # ===== Save ONLY best model =====
        if test_acc > best_acc:
            best_acc = test_acc

            if isinstance(model, ViTSmallBaseline):
                model_name = "vitsmall_baseline"
            elif isinstance(model, MagnitudePruneViTSmall):
                model_name = "vitsmall_magprune"
            elif isinstance(model, AttentionPruneViTSmall):
                model_name = "vitsmall_attnprune"
            else:
                model_name = "vitsmall_unknown"

            os.makedirs("checkpoints", exist_ok=True)
            save_path = f"checkpoints/{model_name}_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Saved BEST checkpoint: {save_path}  (Acc = {best_acc:.2f}%)")
        else:
            print(f"No improvement. Best Acc so far: {best_acc:.2f}%")


    print(f"\n===== Total Training Time: {total_train_time/60:.2f} minutes =====")
