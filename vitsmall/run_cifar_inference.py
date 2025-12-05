import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

from Magnitude_based_VitSmall import MagnitudePruneViTSmall
from Attention_based_VitSmall import AttentionPruneViTSmall
from Baseline_VitSmall import ViTSmallBaseline


def get_cifar100_test(batch_size=64):
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        ),
    ])

    test_set = datasets.CIFAR100(
        root="./data",
        train=False,
        download=True,
        transform=transform_test
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return test_loader


def build_model(model_type: str, num_classes: int = 100):
    if model_type == "baseline":
        model = ViTSmallBaseline(num_classes=num_classes)
    elif model_type == "mag":
        model = MagnitudePruneViTSmall(num_classes=num_classes)
    elif model_type == "attn":
        model = AttentionPruneViTSmall(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return model


def evaluate(model,
             loader,
             device="cuda",
             num_warmup_batches=5):
    model.eval()
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    batch_times = []
    total_eval_start = time.time()

    # 预热若干 batch（不计入统计，主要为了 CUDA 稳定）
    warmup_batches = max(0, num_warmup_batches)
    warmup_done = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # warmup，不记录时间
            if warmup_done < warmup_batches:
                _ = model(images)
                warmup_done += 1
                continue

            # 计时：注意 GPU 要同步
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.time()

            outputs = model(images)

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            end = time.time()

            batch_time = end - start
            batch_times.append(batch_time)

            loss = criterion(outputs, targets)
            total_loss += loss.item()

            _, pred = outputs.max(1)
            total_samples += targets.size(0)
            total_correct += pred.eq(targets).sum().item()

            if batch_idx % 50 == 0:
                print(f"[Eval] Batch {batch_idx}/{len(loader)} "
                      f"Batch time={batch_time:.4f}s  "
                      f"Throughput={targets.size(0)/batch_time:.1f} img/s")

    total_eval_time = time.time() - total_eval_start

    top1_acc = 100.0 * total_correct / total_samples
    avg_loss = total_loss / len(loader)

    batch_times = np.array(batch_times)
    avg_batch_time = batch_times.mean()
    std_batch_time = batch_times.std()
    p50 = np.percentile(batch_times, 50)
    p90 = np.percentile(batch_times, 90)
    p99 = np.percentile(batch_times, 99)

    # 每张图平均 latency
    batch_size = loader.batch_size
    avg_latency_per_image = avg_batch_time / batch_size
    images_per_sec = batch_size / avg_batch_time

    results = {
        "top1_acc": top1_acc,
        "avg_loss": avg_loss,
        "total_eval_time": total_eval_time,
        "avg_batch_time": avg_batch_time,
        "std_batch_time": std_batch_time,
        "p50_batch_time": p50,
        "p90_batch_time": p90,
        "p99_batch_time": p99,
        "avg_latency_per_image": avg_latency_per_image,
        "throughput_img_per_sec": images_per_sec,
        "num_batches": len(batch_times),
        "num_samples": total_samples,
    }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="baseline",
                        choices=["baseline", "mag", "attn"],
                        help="baseline=无剪枝, mag=幅度剪枝, attn=注意力剪枝")
    parser.add_argument("--weights", type=str, default="",
                        help="可选，指定训练好的权重 .pth 路径")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_warmup_batches", type=int, default=5)
    args = parser.parse_args()

    print("=== Eval Config ===")
    print(f"Model Type      : {args.model_type}")
    print(f"Batch Size      : {args.batch_size}")
    print(f"Device          : {args.device}")
    print(f"Warmup Batches  : {args.num_warmup_batches}")
    print(f"Weights         : {args.weights if args.weights else 'ImageNet pretrained / default'}")

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"

    # 1. 构建模型
    model = build_model(args.model_type, num_classes=100)

    # 2. 加载权重（如果给了）
    if args.weights:
        state = torch.load(args.weights, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded weights from: {args.weights}")

    # 3. Dataloader
    test_loader = get_cifar100_test(batch_size=args.batch_size)

    # 4. 评估
    results = evaluate(model, test_loader, device=device,
                       num_warmup_batches=args.num_warmup_batches)

    print("\n===== Eval Results =====")
    print(f"Top-1 Accuracy            : {results['top1_acc']:.2f}%")
    print(f"Avg Loss                  : {results['avg_loss']:.4f}")
    print(f"Total Eval Time           : {results['total_eval_time']:.2f} s")
    print(f"Avg Batch Time            : {results['avg_batch_time']:.4f} s")
    print(f"Batch Time Std            : {results['std_batch_time']:.4f} s")
    print(f"p50 Batch Time            : {results['p50_batch_time']:.4f} s")
    print(f"p90 Batch Time            : {results['p90_batch_time']:.4f} s")
    print(f"p99 Batch Time            : {results['p99_batch_time']:.4f} s")
    print(f"Avg Latency per Image     : {results['avg_latency_per_image']*1000:.4f} ms/img")
    print(f"Throughput                : {results['throughput_img_per_sec']:.2f} img/s")
    print(f"Num Batches               : {results['num_batches']}")
    print(f"Num Samples               : {results['num_samples']}")
    print("=========================\n")


if __name__ == "__main__":
    main()
