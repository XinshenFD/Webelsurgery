#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
晶状体脱位程度图片预测工具

功能：
  1. 预测单张图片的脱位程度（轻/中/重）
  2. 批量预测一个文件夹中的所有图片
  3. 批量预测结果导出为Excel文件

使用方法：
  python predict.py --image 图片路径.jpg
  python predict.py --folder 图片文件夹路径
  python predict.py --folder 图片文件夹路径 --output 结果.xlsx
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd

from model import ResNet50Classifier

# ===================== 配置 =====================
IMAGE_SIZE = 224
NUM_CLASSES = 3
LABEL_MAP = {0: '轻', 1: '中', 2: '重'}
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG'}

# ImageNet 标准化参数
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def get_default_model_path():
    """获取默认模型权重路径"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'weights', 'best_model.pth')


def load_model(model_path, device):
    """加载训练好的模型权重"""
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在 - {model_path}")
        print("请将模型权重文件 best_model.pth 放置到 weights/ 目录下")
        sys.exit(1)

    model = ResNet50Classifier(num_classes=NUM_CLASSES, input_size=IMAGE_SIZE, pretrained=False)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"模型加载成功: {model_path}")
    if 'best_val_acc' in checkpoint:
        print(f"模型验证集准确率: {checkpoint['best_val_acc']:.4f}")

    return model


def get_transform():
    """获取图片预处理变换（与训练时验证集一致）"""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])


def predict_image(model, image_path, transform, device):
    """
    预测单张图片的脱位程度

    Returns:
        pred_label: 预测标签（轻/中/重）
        pred_class: 预测类别编号（0/1/2）
        probs: 各类别概率字典
    """
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)[0]
        pred_class = output.argmax(dim=1).item()

    pred_label = LABEL_MAP[pred_class]
    probs = {LABEL_MAP[i]: f"{probabilities[i].item():.4f}" for i in range(NUM_CLASSES)}

    return pred_label, pred_class, probs


def predict_single(model, image_path, transform, device):
    """预测单张图片并打印结果"""
    if not os.path.exists(image_path):
        print(f"错误：图片文件不存在 - {image_path}")
        return

    print(f"\n预测图片: {image_path}")
    print("-" * 50)

    pred_label, pred_class, probs = predict_image(model, image_path, transform, device)

    print(f"预测结果: {pred_label}")
    print(f"\n各类别概率:")
    for label, prob in probs.items():
        print(f"  {label}: {prob}")


def predict_folder(model, folder_path, transform, device, output_path=None):
    """批量预测文件夹中的所有图片"""
    if not os.path.isdir(folder_path):
        print(f"错误：文件夹不存在 - {folder_path}")
        return

    # 收集所有图片文件
    image_files = []
    for filename in sorted(os.listdir(folder_path)):
        ext = os.path.splitext(filename)[1]
        if ext in SUPPORTED_EXTENSIONS:
            image_files.append(filename)

    if not image_files:
        print(f"文件夹中未找到图片文件: {folder_path}")
        return

    print(f"\n找到 {len(image_files)} 张图片，开始预测...")
    print("-" * 60)

    results = []
    for i, filename in enumerate(image_files):
        image_path = os.path.join(folder_path, filename)
        try:
            pred_label, pred_class, probs = predict_image(model, image_path, transform, device)
            results.append({
                '图片名称': filename,
                '预测结果': pred_label,
                '轻度概率': probs['轻'],
                '中度概率': probs['中'],
                '重度概率': probs['重']
            })
            print(f"  [{i+1}/{len(image_files)}] {filename} → {pred_label}")
        except Exception as e:
            print(f"  [{i+1}/{len(image_files)}] {filename} → 预测失败: {str(e)}")
            results.append({
                '图片名称': filename,
                '预测结果': '预测失败',
                '轻度概率': '',
                '中度概率': '',
                '重度概率': ''
            })

    # 统计
    print("\n" + "=" * 60)
    print("预测统计:")
    df = pd.DataFrame(results)
    valid = df[df['预测结果'] != '预测失败']
    for label in ['轻', '中', '重']:
        count = (valid['预测结果'] == label).sum()
        print(f"  {label}: {count} 张 ({count/len(valid)*100:.1f}%)" if len(valid) > 0 else f"  {label}: 0 张")

    # 保存结果
    if output_path is None:
        output_path = os.path.join(folder_path, '预测结果.xlsx')

    df.to_excel(output_path, index=False)
    print(f"\n结果已保存到: {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='晶状体脱位程度图片预测工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  预测单张图片:
    python predict.py --image test.jpg

  批量预测文件夹:
    python predict.py --folder ./images

  指定输出文件:
    python predict.py --folder ./images --output results.xlsx

  使用CPU:
    python predict.py --image test.jpg --device cpu

  指定模型路径:
    python predict.py --image test.jpg --model ./weights/best_model.pth
        """
    )

    parser.add_argument('--image', type=str, help='单张图片路径')
    parser.add_argument('--folder', type=str, help='图片文件夹路径（批量预测）')
    parser.add_argument('--model', type=str, default=None, help='模型权重路径（默认: weights/best_model.pth）')
    parser.add_argument('--output', type=str, default=None, help='批量预测结果输出路径（默认: 文件夹内/预测结果.xlsx）')
    parser.add_argument('--device', type=str, default=None, help='运行设备: cuda 或 cpu（默认: 自动检测）')

    args = parser.parse_args()

    # 参数校验
    if args.image is None and args.folder is None:
        parser.print_help()
        print("\n错误：请指定 --image 或 --folder 参数")
        sys.exit(1)

    # 设备配置
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 模型路径
    model_path = args.model if args.model else get_default_model_path()

    # 加载模型
    model = load_model(model_path, device)
    transform = get_transform()

    # 执行预测
    if args.image:
        predict_single(model, args.image, transform, device)
    elif args.folder:
        predict_folder(model, args.folder, transform, device, args.output)


if __name__ == '__main__':
    main()
