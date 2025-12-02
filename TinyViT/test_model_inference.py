"""
模型推理测试脚本
输入：模型权重文件(.pth)和测试图片
输出：分类结果、正确率、推理速度
"""

import torch
import torch.nn as nn
import time
import numpy as np
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.tiny_vit import tiny_vit_5m_224
from tqdm import tqdm
import os


def get_test_loader(data_root, batch_size=64):
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
        download=False,
        transform=transform_test,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return test_loader


def load_model(weight_path, num_classes=100, use_pruning=False, pruning_ratio=0.2, pruning_method='attention', device=None):
    """
    加载模型
    
    Args:
        weight_path: 模型权重文件路径
        num_classes: 类别数
        use_pruning: 是否使用token pruning
        pruning_ratio: pruning比例（如果use_pruning=True）
        pruning_method: pruning方法（如果use_pruning=True）
        device: 目标设备
    
    Returns:
        model: 加载好的模型
    """
    print(f"加载模型权重: {weight_path}")
    
    if use_pruning:
        print(f"使用 Token Pruning (ratio={pruning_ratio}, method={pruning_method})")
        model = tiny_vit_5m_224(
            pretrained=False,
            num_classes=num_classes,
            token_pruning_ratio=pruning_ratio,
            token_pruning_method=pruning_method
        )
    else:
        model = tiny_vit_5m_224(
            pretrained=False,
            num_classes=num_classes
        )
    
    # 加载权重（先加载到 CPU）
    state_dict = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    
    # 移动到目标设备
    if device is not None:
        model = model.to(device)
        # 递归确保所有子模块的 buffer 都在正确的设备上
        def move_buffers_to_device(module, target_device):
            for name, buffer in module.named_buffers(recurse=False):
                if buffer.device != target_device:
                    buffer.data = buffer.data.to(target_device)
            for child in module.children():
                move_buffers_to_device(child, target_device)
        
        move_buffers_to_device(model, device)
    
    print("模型加载成功！")
    
    return model


def evaluate_accuracy_and_speed(model, test_loader, device, num_warmup=10):
    """
    评估模型准确率和推理速度
    
    Args:
        model: 要测试的模型
        test_loader: 测试数据加载器
        device: 设备 (cuda/cpu)
        num_warmup: 预热轮数
    
    Returns:
        dict: 包含准确率和速度统计信息
    """
    model.eval()
    # 确保模型在正确的设备上
    if next(model.parameters()).device != device:
        model = model.to(device)
        # 递归确保所有子模块的 buffer 都在正确的设备上
        def move_buffers_to_device(module, target_device):
            for name, buffer in module.named_buffers(recurse=False):
                if buffer.device != target_device:
                    buffer.data = buffer.data.to(target_device)
            for child in module.children():
                move_buffers_to_device(child, target_device)
        
        move_buffers_to_device(model, device)
    
    # 预热
    print(f"\n预热中... ({num_warmup} 轮)")
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
    print("开始评估...")
    correct = 0
    total = 0
    times = []
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="测试进度"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
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
            
            # 计算准确率
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 保存预测结果
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # 记录时间
            times.append(elapsed_time)
    
    # 计算统计信息
    times = np.array(times)
    accuracy = 100.0 * correct / total
    total_time = times.sum()
    avg_time_per_batch = times.mean()
    std_time_per_batch = times.std()
    min_time_per_batch = times.min()
    max_time_per_batch = times.max()
    avg_time_per_sample = avg_time_per_batch / batch_size
    throughput = total / total_time  # samples per second
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'total_time': total_time,
        'avg_time_per_batch': avg_time_per_batch,
        'std_time_per_batch': std_time_per_batch,
        'min_time_per_batch': min_time_per_batch,
        'max_time_per_batch': max_time_per_batch,
        'avg_time_per_sample': avg_time_per_sample,
        'throughput': throughput,
        'predictions': all_predictions,
        'targets': all_targets,
    }


def print_results(results, model_name="模型"):
    """打印测试结果"""
    print("\n" + "=" * 80)
    print(f"{model_name} 测试结果")
    print("=" * 80)
    
    print("\n【准确率】")
    print(f"  正确数: {results['correct']}/{results['total']}")
    print(f"  准确率: {results['accuracy']:.2f}%")
    
    print("\n【推理速度】")
    print(f"  总测试时间: {results['total_time']:.2f} 秒")
    print(f"  平均每批次时间: {results['avg_time_per_batch']:.4f} 秒")
    print(f"  平均每个样本时间: {results['avg_time_per_sample']*1000:.4f} 毫秒")
    print(f"  吞吐量: {results['throughput']:.2f} samples/秒")
    print(f"  最小批次时间: {results['min_time_per_batch']:.4f} 秒")
    print(f"  最大批次时间: {results['max_time_per_batch']:.4f} 秒")
    print(f"  标准差: {results['std_time_per_batch']:.4f} 秒")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='测试模型推理速度和准确率')
    parser.add_argument('--weight', type=str, required=True,
                        help='模型权重文件路径 (.pth)')
    parser.add_argument('--data_root', type=str,
                        default=r"C:\Users\lmh98\Desktop\CV Project\data\cifar-100-python",
                        help='测试数据路径')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小 (默认: 64)')
    parser.add_argument('--num_classes', type=int, default=100,
                        help='类别数 (默认: 100)')
    parser.add_argument('--use_pruning', action='store_true',
                        help='是否使用 Token Pruning')
    parser.add_argument('--pruning_ratio', type=float, default=0.2,
                        help='Token Pruning 比例 (默认: 0.2)')
    parser.add_argument('--pruning_method', type=str, default='attention',
                        choices=['attention', 'magnitude'],
                        help='Token Pruning 方法 (默认: attention)')
    parser.add_argument('--num_warmup', type=int, default=10,
                        help='预热轮数 (默认: 10)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='设备选择 (默认: auto)')
    
    args = parser.parse_args()
    
    # 检查权重文件是否存在
    if not os.path.exists(args.weight):
        print(f"错误: 找不到权重文件 {args.weight}")
        return
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("=" * 80)
    print("模型推理测试")
    print("=" * 80)
    print(f"权重文件: {args.weight}")
    print(f"数据路径: {args.data_root}")
    print(f"设备: {device}")
    print(f"批次大小: {args.batch_size}")
    print(f"类别数: {args.num_classes}")
    if args.use_pruning:
        print(f"Token Pruning: 启用 (ratio={args.pruning_ratio}, method={args.pruning_method})")
    else:
        print(f"Token Pruning: 未启用")
    print()
    
    # 加载测试数据
    print("加载测试数据...")
    try:
        test_loader = get_test_loader(args.data_root, batch_size=args.batch_size)
        print(f"测试数据加载完成，共 {len(test_loader.dataset)} 个样本")
    except Exception as e:
        print(f"错误: 无法加载测试数据 - {e}")
        return
    
    # 加载模型
    try:
        model = load_model(
            args.weight,
            num_classes=args.num_classes,
            use_pruning=args.use_pruning,
            pruning_ratio=args.pruning_ratio,
            pruning_method=args.pruning_method,
            device=device
        )
    except Exception as e:
        print(f"错误: 无法加载模型 - {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 评估模型
    try:
        results = evaluate_accuracy_and_speed(
            model, test_loader, device, num_warmup=args.num_warmup
        )
        
        # 打印结果
        model_name = os.path.basename(args.weight)
        print_results(results, model_name)
        
        # 可选：保存详细结果到文件
        output_file = args.weight.replace('.pth', '_test_results.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"模型: {args.weight}\n")
            f.write(f"设备: {device}\n")
            f.write(f"批次大小: {args.batch_size}\n")
            f.write(f"\n准确率: {results['accuracy']:.2f}%\n")
            f.write(f"正确数: {results['correct']}/{results['total']}\n")
            f.write(f"\n推理速度:\n")
            f.write(f"  总时间: {results['total_time']:.2f} 秒\n")
            f.write(f"  平均每批次: {results['avg_time_per_batch']:.4f} 秒\n")
            f.write(f"  平均每样本: {results['avg_time_per_sample']*1000:.4f} 毫秒\n")
            f.write(f"  吞吐量: {results['throughput']:.2f} samples/秒\n")
        
        print(f"\n详细结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"错误: 评估过程中出错 - {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n测试完成！")


if __name__ == "__main__":
    main()

