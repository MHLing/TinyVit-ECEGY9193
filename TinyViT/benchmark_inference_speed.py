"""
推理速度对比脚本
比较原始模型和带 Token Pruning 模型的推理速度
"""

import torch
import torch.nn as nn
import time
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.tiny_vit import tiny_vit_5m_224
from tqdm import tqdm


def get_test_loader(data_root, batch_size=64, num_samples=1000):
    """获取测试数据加载器"""
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761],
        ),
    ])

    test_set = datasets.CIFAR100(
        root=data_root,
        train=False,
        download=False,  # 假设数据已存在
        transform=transform_test,
    )
    
    # 如果指定了 num_samples，只取前 N 个样本
    if num_samples > 0 and num_samples < len(test_set):
        indices = torch.randperm(len(test_set))[:num_samples]
        test_set = torch.utils.data.Subset(test_set, indices)

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return test_loader


def benchmark_model(model, test_loader, device, num_warmup=10, num_runs=100):
    """
    测试模型推理速度
    
    Args:
        model: 要测试的模型
        test_loader: 测试数据加载器
        device: 设备 (cuda/cpu)
        num_warmup: 预热轮数
        num_runs: 正式测试轮数
    
    Returns:
        dict: 包含各种时间统计信息
    """
    model.eval()
    model.to(device)
    
    # 预热
    print(f"  预热中... ({num_warmup} 轮)")
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= num_warmup:
                break
            images = images.to(device, non_blocking=True)
            _ = model(images)
    
    # 同步 GPU（如果使用 CUDA）
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 正式测试
    print(f"  正式测试中... ({num_runs} 轮)")
    times = []
    total_samples = 0
    
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= num_runs:
                break
            
            images = images.to(device, non_blocking=True)
            batch_size = images.shape[0]
            
            # 使用 GPU Event 精确计时（如果使用 CUDA）
            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_time = time.time()
            
            # 推理
            outputs = model(images)
            
            if device.type == 'cuda':
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # 转换为秒
            else:
                elapsed_time = time.time() - start_time
            
            times.append(elapsed_time)
            total_samples += batch_size
    
    # 统计信息
    times = np.array(times)
    total_time = times.sum()
    avg_time_per_batch = times.mean()
    std_time_per_batch = times.std()
    min_time_per_batch = times.min()
    max_time_per_batch = times.max()
    
    # 计算吞吐量
    throughput = total_samples / total_time  # samples per second
    
    return {
        'total_time': total_time,
        'total_samples': total_samples,
        'avg_time_per_batch': avg_time_per_batch,
        'std_time_per_batch': std_time_per_batch,
        'min_time_per_batch': min_time_per_batch,
        'max_time_per_batch': max_time_per_batch,
        'throughput': throughput,
        'avg_time_per_sample': avg_time_per_batch / (total_samples / len(times)),
    }


def evaluate_accuracy(model, test_loader, device):
    """评估模型准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="  评估准确率"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy


def main():
    # ===== 配置 =====
    data_root = r"C:\Users\lmh98\Desktop\CV Project\data\cifar-100-python"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模型权重路径
    model_baseline_path = "tinyvit5m_cifar100_best.pth"  # 原始模型
    model_pruning_path = "tinyvit5m_cifar100_pruning_best.pth"  # Pruning 模型
    
    # 测试配置
    batch_size = 64
    num_warmup = 10
    num_runs = 100
    num_samples_for_accuracy = 1000  # 用于准确率评估的样本数
    
    print("=" * 80)
    print("模型推理速度对比测试")
    print("=" * 80)
    print(f"设备: {device}")
    print(f"批次大小: {batch_size}")
    print(f"预热轮数: {num_warmup}")
    print(f"测试轮数: {num_runs}")
    print()
    
    # ===== 加载测试数据 =====
    print("加载测试数据...")
    test_loader = get_test_loader(data_root, batch_size=batch_size, num_samples=-1)
    test_loader_accuracy = get_test_loader(data_root, batch_size=batch_size, num_samples=num_samples_for_accuracy)
    print(f"测试数据加载完成，共 {len(test_loader.dataset)} 个样本")
    print()
    
    results = {}
    
    # ===== 测试原始模型 =====
    print("=" * 80)
    print("1. 测试原始模型（Baseline）")
    print("=" * 80)
    
    try:
        model_baseline = tiny_vit_5m_224(pretrained=False, num_classes=100)
        if torch.cuda.is_available():
            state_dict = torch.load(model_baseline_path, map_location='cpu')
        else:
            state_dict = torch.load(model_baseline_path, map_location='cpu')
        model_baseline.load_state_dict(state_dict)
        
        print("模型加载成功")
        
        # 测试速度
        print("\n测试推理速度...")
        baseline_stats = benchmark_model(model_baseline, test_loader, device, num_warmup, num_runs)
        results['baseline'] = baseline_stats
        
        # 测试准确率
        print("\n测试准确率...")
        baseline_accuracy = evaluate_accuracy(model_baseline, test_loader_accuracy, device)
        results['baseline']['accuracy'] = baseline_accuracy
        
        print(f"\n原始模型测试完成！")
        print(f"  准确率: {baseline_accuracy:.2f}%")
        
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 {model_baseline_path}")
        print("跳过原始模型测试")
    except Exception as e:
        print(f"错误: {e}")
        print("跳过原始模型测试")
    
    print()
    
    # ===== 测试 Pruning 模型 =====
    print("=" * 80)
    print("2. 测试 Token Pruning 模型")
    print("=" * 80)
    
    try:
        # 加载 pruning 模型（需要指定 pruning 参数）
        token_pruning_ratio = 0.2
        token_pruning_method = 'attention'
        
        model_pruning = tiny_vit_5m_224(
            pretrained=False,
            num_classes=100,
            token_pruning_ratio=token_pruning_ratio,
            token_pruning_method=token_pruning_method
        )
        
        if torch.cuda.is_available():
            state_dict = torch.load(model_pruning_path, map_location='cpu')
        else:
            state_dict = torch.load(model_pruning_path, map_location='cpu')
        model_pruning.load_state_dict(state_dict)
        
        print(f"模型加载成功 (pruning_ratio={token_pruning_ratio}, method={token_pruning_method})")
        
        # 测试速度
        print("\n测试推理速度...")
        pruning_stats = benchmark_model(model_pruning, test_loader, device, num_warmup, num_runs)
        results['pruning'] = pruning_stats
        
        # 测试准确率
        print("\n测试准确率...")
        pruning_accuracy = evaluate_accuracy(model_pruning, test_loader_accuracy, device)
        results['pruning']['accuracy'] = pruning_accuracy
        
        print(f"\nPruning 模型测试完成！")
        print(f"  准确率: {pruning_accuracy:.2f}%")
        
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 {model_pruning_path}")
        print("跳过 Pruning 模型测试")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        print("跳过 Pruning 模型测试")
    
    print()
    
    # ===== 对比结果 =====
    print("=" * 80)
    print("对比结果")
    print("=" * 80)
    
    if 'baseline' in results and 'pruning' in results:
        baseline = results['baseline']
        pruning = results['pruning']
        
        print("\n【推理速度对比】")
        print(f"{'指标':<30} {'原始模型':<20} {'Pruning模型':<20} {'提升':<15}")
        print("-" * 85)
        
        # 平均每批次时间
        baseline_batch_time = baseline['avg_time_per_batch']
        pruning_batch_time = pruning['avg_time_per_batch']
        speedup_batch = baseline_batch_time / pruning_batch_time
        print(f"{'平均每批次时间 (s)':<30} {baseline_batch_time:<20.4f} {pruning_batch_time:<20.4f} {speedup_batch:<15.2f}x")
        
        # 平均每个样本时间
        baseline_sample_time = baseline['avg_time_per_sample']
        pruning_sample_time = pruning['avg_time_per_sample']
        speedup_sample = baseline_sample_time / pruning_sample_time
        print(f"{'平均每个样本时间 (s)':<30} {baseline_sample_time:<20.6f} {pruning_sample_time:<20.6f} {speedup_sample:<15.2f}x")
        
        # 吞吐量
        baseline_throughput = baseline['throughput']
        pruning_throughput = pruning['throughput']
        throughput_improvement = (pruning_throughput / baseline_throughput - 1) * 100
        print(f"{'吞吐量 (samples/s)':<30} {baseline_throughput:<20.2f} {pruning_throughput:<20.2f} {throughput_improvement:+.2f}%")
        
        # 总时间
        baseline_total = baseline['total_time']
        pruning_total = pruning['total_time']
        time_saved = baseline_total - pruning_total
        time_saved_percent = (time_saved / baseline_total) * 100
        print(f"{'总测试时间 (s)':<30} {baseline_total:<20.2f} {pruning_total:<20.2f} {time_saved_percent:+.2f}%")
        
        print("\n【准确率对比】")
        baseline_acc = baseline['accuracy']
        pruning_acc = pruning['accuracy']
        acc_drop = baseline_acc - pruning_acc
        print(f"原始模型准确率: {baseline_acc:.2f}%")
        print(f"Pruning 模型准确率: {pruning_acc:.2f}%")
        print(f"准确率下降: {acc_drop:.2f}%")
        
        print("\n【详细统计】")
        print(f"\n原始模型:")
        print(f"  最小批次时间: {baseline['min_time_per_batch']:.4f}s")
        print(f"  最大批次时间: {baseline['max_time_per_batch']:.4f}s")
        print(f"  标准差: {baseline['std_time_per_batch']:.4f}s")
        
        print(f"\nPruning 模型:")
        print(f"  最小批次时间: {pruning['min_time_per_batch']:.4f}s")
        print(f"  最大批次时间: {pruning['max_time_per_batch']:.4f}s")
        print(f"  标准差: {pruning['std_time_per_batch']:.4f}s")
        
        # 总结
        print("\n" + "=" * 80)
        print("总结")
        print("=" * 80)
        print(f"✅ 速度提升: {speedup_sample:.2f}x")
        print(f"✅ 吞吐量提升: {throughput_improvement:+.2f}%")
        print(f"✅ 准确率下降: {acc_drop:.2f}%")
        print(f"✅ 速度-精度权衡: 用 {acc_drop:.2f}% 的准确率换取了 {speedup_sample:.2f}x 的速度提升")
        
    elif 'baseline' in results:
        print("只有原始模型测试结果:")
        baseline = results['baseline']
        print(f"  平均每批次时间: {baseline['avg_time_per_batch']:.4f}s")
        print(f"  吞吐量: {baseline['throughput']:.2f} samples/s")
        print(f"  准确率: {baseline['accuracy']:.2f}%")
        
    elif 'pruning' in results:
        print("只有 Pruning 模型测试结果:")
        pruning = results['pruning']
        print(f"  平均每批次时间: {pruning['avg_time_per_batch']:.4f}s")
        print(f"  吞吐量: {pruning['throughput']:.2f} samples/s")
        print(f"  准确率: {pruning['accuracy']:.2f}%")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

