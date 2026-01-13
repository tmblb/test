#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集准备工具

功能包括：
1. 创建数据集目录结构
2. 下载公开数据集（COCO2017子集）
3. 数据标注（使用预训练模型自动标注示例）
4. 数据集划分（训练集/验证集/测试集）
5. 生成YOLO格式的数据集配置文件
"""

import os
import sys
import shutil
import random
import requests
import argparse
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


def create_directory_structure(root_dir="data"):
    """
    创建数据集目录结构
    
    Args:
        root_dir: 数据集根目录
    """
    print(f"创建数据集目录结构...")
    
    # 创建主目录
    os.makedirs(root_dir, exist_ok=True)
    
    # 创建子目录
    directories = [
        "images/train",
        "images/val", 
        "images/test",
        "labels/train",
        "labels/val",
        "labels/test"
    ]
    
    for dir_path in directories:
        full_path = os.path.join(root_dir, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"  创建目录: {full_path}")
    
    print(f"目录结构创建完成！")


class COCODatasetDownloader:
    """
    COCO2017数据集下载器（只下载部分数据用于示例）
    """
    
    def __init__(self, save_dir="data"):
        self.save_dir = save_dir
        self.base_url = "http://images.cocodataset.org"
        self.categories = {
            "person": 1,
            "car": 3,
            "dog": 18
        }
    
    def download_sample_images(self, num_images=50):
        """
        下载COCO2017样本图像
        
        Args:
            num_images: 下载图像数量
        """
        print(f"下载COCO2017样本图像...")
        
        # 创建临时目录
        temp_dir = os.path.join(self.save_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 下载图像
        images_downloaded = 0
        for img_id in tqdm(range(1, num_images + 1000), desc="下载图像"):
            if images_downloaded >= num_images:
                break
            
            img_url = f"{self.base_url}/train2017/000000{img_id:06d}.jpg"
            img_path = os.path.join(temp_dir, f"{img_id}.jpg")
            
            try:
                response = requests.get(img_url, timeout=10)
                if response.status_code == 200:
                    with open(img_path, "wb") as f:
                        f.write(response.content)
                    images_downloaded += 1
            except Exception:
                continue
        
        print(f"成功下载 {images_downloaded} 张图像！")
        return temp_dir


class AutoLabeler:
    """
    使用预训练YOLOv8模型自动标注图像
    """
    
    def __init__(self, model_path="yolov8s.pt", confidence_threshold=0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
    
    def label_image(self, image_path, label_path):
        """
        标注单张图像并保存为YOLO格式
        
        Args:
            image_path: 图像路径
            label_path: 标注文件保存路径
        """
        # 进行推理
        results = self.model(image_path, conf=self.confidence_threshold)
        
        # 获取图像尺寸
        img = results[0].orig_img
        h, w = img.shape[:2]
        
        # 准备标注内容
        label_lines = []
        for box in results[0].boxes:
            # 获取类别和边界框
            cls = int(box.cls[0])
            xyxy = box.xyxy[0].tolist()
            
            # 转换为YOLO格式（归一化坐标）
            x_center = (xyxy[0] + xyxy[2]) / 2 / w
            y_center = (xyxy[1] + xyxy[3]) / 2 / h
            width = (xyxy[2] - xyxy[0]) / w
            height = (xyxy[3] - xyxy[1]) / h
            
            # 只保留person, car, dog三个类别（COCO ID: 0, 2, 16）
            if cls in [0, 2, 16]:
                # 转换为我们自定义的类别ID（0: person, 1: car, 2: dog）
                custom_cls = {0: 0, 2: 1, 16: 2}[cls]
                label_lines.append(f"{custom_cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # 保存标注文件
        if label_lines:
            with open(label_path, "w") as f:
                f.write("\n".join(label_lines) + "\n")
            return True
        else:
            return False
    
    def label_directory(self, image_dir, label_dir):
        """
        批量标注目录中的图像
        
        Args:
            image_dir: 图像目录
            label_dir: 标注文件保存目录
        """
        print(f"使用YOLOv8模型自动标注图像...")
        
        # 获取所有图像文件
        image_files = [f for f in os.listdir(image_dir) 
                     if f.endswith((".jpg", ".jpeg", ".png"))]
        
        valid_images = 0
        for image_file in tqdm(image_files, desc="标注图像"):
            image_path = os.path.join(image_dir, image_file)
            label_file = os.path.splitext(image_file)[0] + ".txt"
            label_path = os.path.join(label_dir, label_file)
            
            if self.label_image(image_path, label_path):
                valid_images += 1
        
        print(f"成功标注 {valid_images} 张图像！")
        return valid_images


class DatasetSplitter:
    """
    数据集划分工具（训练集/验证集/测试集）
    """
    
    def __init__(self, root_dir="data", train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        self.root_dir = root_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # 验证比例之和为1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "比例之和必须为1"
    
    def split(self, source_dir):
        """
        划分数据集
        
        Args:
            source_dir: 原始图像目录
        """
        print(f"划分数据集...")
        
        # 获取所有图像文件
        image_files = [f for f in os.listdir(source_dir) 
                     if f.endswith((".jpg", ".jpeg", ".png"))]
        
        # 打乱文件列表
        random.shuffle(image_files)
        
        # 计算划分数量
        total = len(image_files)
        train_count = int(total * self.train_ratio)
        val_count = int(total * self.val_ratio)
        test_count = total - train_count - val_count
        
        print(f"\n数据集划分情况：")
        print(f"  总图像数: {total}")
        print(f"  训练集: {train_count} ({self.train_ratio*100}%)")
        print(f"  验证集: {val_count} ({self.val_ratio*100}%)")
        print(f"  测试集: {test_count} ({self.test_ratio*100}%)")
        
        # 执行划分
        self._copy_files(image_files[:train_count], "train")
        self._copy_files(image_files[train_count:train_count+val_count], "val")
        self._copy_files(image_files[train_count+val_count:], "test")
        
        print(f"\n数据集划分完成！")
    
    def _copy_files(self, file_list, split_type):
        """
        复制文件到指定的划分目录
        
        Args:
            file_list: 文件列表
            split_type: 划分类型（train/val/test）
        """
        for image_file in tqdm(file_list, desc=f"复制{split_type}集"):
            print(image_file)
            # 复制图像文件
            src_image = os.path.join(self.root_dir, "temp", image_file)
            dst_image = os.path.join(self.root_dir, "images", split_type, image_file)
            shutil.copy(src_image, dst_image)
            
            # 复制标注文件
            label_file = os.path.splitext(image_file)[0] + ".txt"
            src_label = os.path.join(self.root_dir, "temp_labels", label_file)
            dst_label = os.path.join(self.root_dir, "labels", split_type, label_file)
            
            if os.path.exists(src_label):
                shutil.copy(src_label, dst_label)


class DatasetConfigGenerator:
    """
    生成YOLO格式的数据集配置文件
    """
    
    def __init__(self, root_dir="data"):
        self.root_dir = root_dir
    
    def generate_config(self, names):
        """
        生成配置文件
        
        Args:
            names: 类别名称列表
        """
        print(f"生成数据集配置文件...")
        
        # 创建配置内容
        config_content = f"""path: {self.root_dir}  # 数据集根目录

train: images/train  # 训练图像目录
val: images/val      # 验证图像目录
test: images/test    # 测试图像目录

# 类别名称
names:
{chr(10).join([f'  {i}: {name}' for i, name in enumerate(names)])}
"""
        # 保存配置文件
        config_path = os.path.join(self.root_dir, "dataset.yaml")
        with open(config_path, "w") as f:
            f.write(config_content)
        
        print(f"配置文件已保存到: {config_path}")
        print("\n配置文件内容:")
        print(config_content)
        
        return config_path


def main(args):
    """
    主函数
    """
    # 1. 创建目录结构
    create_directory_structure(args.root_dir)
    
    # 2. 下载样本图像
    downloader = COCODatasetDownloader(args.root_dir)
    temp_dir = downloader.download_sample_images(args.num_images)
    
    # 3. 自动标注图像
    labeler = AutoLabeler(confidence_threshold=args.confidence)
    temp_labels_dir = os.path.join(args.root_dir, "temp_labels")
    os.makedirs(temp_labels_dir, exist_ok=True)
    labeler.label_directory(temp_dir, temp_labels_dir)
    
    # 4. 划分数据集
    splitter = DatasetSplitter(
        args.root_dir, 
        args.train_ratio, 
        args.val_ratio, 
        args.test_ratio
    )
    splitter.split(temp_dir)
    
    # 5. 生成配置文件
    config_generator = DatasetConfigGenerator(args.root_dir)
    config_generator.generate_config(["person", "car", "dog"])
    
    # 6. 清理临时文件
    print(f"清理临时文件...")
    shutil.rmtree(temp_dir)
    shutil.rmtree(temp_labels_dir)
    
    print(f"\n数据集准备完成！")
    print(f"\n数据集信息:")
    print(f"  根目录: {args.root_dir}")
    print(f"  图像数量: {args.num_images}")
    print(f"  类别: person, car, dog")
    print(f"\n下一步操作:")
    print(f"  1. 检查数据集中的图像和标注是否正确")
    print(f"  2. 使用以下命令训练模型:")
    print(f"     python yolo_train.py")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="数据集准备工具")
    parser.add_argument("--root_dir", type=str, default="data", help="数据集根目录")
    parser.add_argument("--num_images", type=int, default=500, help="下载图像数量")
    parser.add_argument("--confidence", type=float, default=0.5, help="自动标注的置信度阈值")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="测试集比例")
    
    args = parser.parse_args()
    
    # 检查比例参数
    if not (0 < args.train_ratio < 1 and 0 < args.val_ratio < 1 and 0 < args.test_ratio < 1):
        print("错误：比例参数必须在0到1之间")
        sys.exit(1)
    
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) >= 0.001:
        print("错误：比例之和必须为1")
        sys.exit(1)
    
    # 运行主函数
    main(args)
