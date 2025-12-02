import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, writer=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 使用 GPU Event 来精确测量时间
    use_cuda = device.type == 'cuda'
    if use_cuda:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        batch_times = []
    
    # 使用 tqdm 显示进度条
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]', ncols=120)
    
    for i, (images, targets) in enumerate(pbar):
        if use_cuda:
            start_event.record()
        
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if use_cuda:
            end_event.record()
            torch.cuda.synchronize()
            batch_time = start_event.elapsed_time(end_event) / 1000.0  # 转换为秒
            batch_times.append(batch_time)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 更新进度条信息
        current_acc = correct / total * 100.0
        current_loss = running_loss / total
        
        if use_cuda and len(batch_times) > 0:
            avg_batch_time = sum(batch_times) / len(batch_times)
            remaining_batches = len(loader) - (i + 1)
            eta_seconds = avg_batch_time * remaining_batches
            eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%',
                'Time/batch': f'{avg_batch_time:.3f}s',
                'ETA': eta_str
            })
        else:
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%'
            })

    epoch_loss = running_loss / total
    epoch_acc = correct / total * 100.0
    
    # 记录到 TensorBoard
    if writer is not None:
        writer.add_scalar('Train/Loss', epoch_loss, epoch)
        writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
        if use_cuda and len(batch_times) > 0:
            avg_batch_time = sum(batch_times) / len(batch_times)
            writer.add_scalar('Train/Time_per_batch', avg_batch_time, epoch)
    
    print(f"[Train] Epoch {epoch}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%", end='')
    if use_cuda and len(batch_times) > 0:
        avg_batch_time = sum(batch_times) / len(batch_times)
        total_time = sum(batch_times)
        print(f", Avg Time/batch={avg_batch_time:.3f}s, Total Time={total_time:.2f}s")
    else:
        print()
    
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device, epoch, writer=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 使用 GPU Event 来精确测量时间
    use_cuda = device.type == 'cuda'
    if use_cuda:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        batch_times = []

    # 使用 tqdm 显示进度条
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Eval ]', ncols=120)

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(pbar):
            if use_cuda:
                start_event.record()
            
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)
            
            if use_cuda:
                end_event.record()
                torch.cuda.synchronize()
                batch_time = start_event.elapsed_time(end_event) / 1000.0  # 转换为秒
                batch_times.append(batch_time)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 更新进度条信息
            current_acc = correct / total * 100.0
            current_loss = running_loss / total
            
            if use_cuda and len(batch_times) > 0:
                avg_batch_time = sum(batch_times) / len(batch_times)
                remaining_batches = len(loader) - (batch_idx + 1)
                eta_seconds = avg_batch_time * remaining_batches
                eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
                pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.2f}%',
                    'Time/batch': f'{avg_batch_time:.3f}s',
                    'ETA': eta_str
                })
            else:
                pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })

    epoch_loss = running_loss / total
    epoch_acc = correct / total * 100.0
    
    # 记录到 TensorBoard
    if writer is not None:
        writer.add_scalar('Val/Loss', epoch_loss, epoch)
        writer.add_scalar('Val/Accuracy', epoch_acc, epoch)
        if use_cuda and len(batch_times) > 0:
            avg_batch_time = sum(batch_times) / len(batch_times)
            writer.add_scalar('Val/Time_per_batch', avg_batch_time, epoch)
    
    print(f"[Eval ] Epoch {epoch}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%", end='')
    if use_cuda and len(batch_times) > 0:
        avg_batch_time = sum(batch_times) / len(batch_times)
        total_time = sum(batch_times)
        print(f", Avg Time/batch={avg_batch_time:.3f}s, Total Time={total_time:.2f}s")
    else:
        print()
    
    return epoch_loss, epoch_acc


def main():
    # ===== 基本配置 =====
    data_root = r"C:\Users\lmh98\Desktop\CV Project\data\cifar-100-python"   # 数据会下在 TinyViT/cifar100_data 下面
    batch_size = 64
    epochs = 50                     # 先跑 50 个 epoch 看看情况
    lr = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ===== TensorBoard Writer =====
    writer = SummaryWriter(log_dir='runs/tinyvit5m_cifar100_pruning')

    # ===== 数据加载器 =====
    train_loader, test_loader = get_dataloaders(data_root, batch_size=batch_size)

    # ===== 构建 TinyViT-5M 模型 =====
    # num_classes=100 对应 CIFAR-100
    # Token pruning 配置
    token_pruning_ratio = 0.2  # 剪枝 20% 的 tokens (可调整: 0.0-0.5)
    token_pruning_method = 'magnitude'  # 'attention' 或 'magnitude'
    
    model = tiny_vit_5m_224(
        pretrained=True, 
        num_classes=100,
        token_pruning_ratio=token_pruning_ratio,
        token_pruning_method=token_pruning_method
    )
    model.to(device)
    
    print(f"Token Pruning enabled: ratio={token_pruning_ratio}, method={token_pruning_method}")

    # ===== 损失函数 & 优化器 =====
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    best_acc = 0.0

    # 使用 GPU Event 来精确测量总时间
    use_cuda = device.type == 'cuda'
    if use_cuda:
        total_start_event = torch.cuda.Event(enable_timing=True)
        total_end_event = torch.cuda.Event(enable_timing=True)
        total_start_event.record()
    else:
        total_start = time.time()

    for epoch in range(1, epochs + 1):
        # 使用 GPU Event 来精确测量 epoch 时间
        if use_cuda:
            epoch_start_event = torch.cuda.Event(enable_timing=True)
            epoch_end_event = torch.cuda.Event(enable_timing=True)
            epoch_start_event.record()
        else:
            epoch_start = time.time()
        
        train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)
        _, val_acc = evaluate(model, test_loader, criterion, device, epoch, writer)

        # 简单保存最好 ckpt
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "tinyvit5m_cifar100_pruning_best.pth")
            print(f"*** New best acc: {best_acc:.2f}%, checkpoint saved.")

        # 计算 epoch 时间
        if use_cuda:
            epoch_end_event.record()
            torch.cuda.synchronize()
            epoch_time = epoch_start_event.elapsed_time(epoch_end_event) / 1000.0  # 转换为秒
            writer.add_scalar('Time/Epoch_time', epoch_time, epoch)
            print(f"Epoch {epoch} time: {epoch_time:.2f}s")
        else:
            epoch_time = time.time() - epoch_start
            writer.add_scalar('Time/Epoch_time', epoch_time, epoch)
            print(f"Epoch {epoch} time: {epoch_time:.2f}s")
        
        # 计算剩余时间（ETA）
        if epoch < epochs:
            remaining_epochs = epochs - epoch
            if use_cuda:
                # 使用已完成的 epoch 平均时间估算
                # 这里简化处理，实际可以记录所有 epoch 时间然后平均
                estimated_remaining = epoch_time * remaining_epochs
            else:
                estimated_remaining = epoch_time * remaining_epochs
            eta_hours = int(estimated_remaining // 3600)
            eta_mins = int((estimated_remaining % 3600) // 60)
            eta_secs = int(estimated_remaining % 60)
            print(f"Estimated remaining time: {eta_hours}h {eta_mins}m {eta_secs}s")
        
        print()
    
    # 计算总训练时间
    if use_cuda:
        total_end_event.record()
        torch.cuda.synchronize()
        total_time = total_start_event.elapsed_time(total_end_event) / 1000.0  # 转换为秒
    else:
        total_time = time.time() - total_start
    
    total_hours = int(total_time // 3600)
    total_mins = int((total_time % 3600) // 60)
    total_secs = int(total_time % 60)
    
    print("=" * 60)
    print(f"Training finished!")
    print(f"Best Acc: {best_acc:.2f}%")
    print(f"Total training time: {total_hours}h {total_mins}m {total_secs}s ({total_time:.2f}s)")
    print("=" * 60)
    
    # 关闭 TensorBoard writer
    writer.close()


if __name__ == "__main__":
    main()
