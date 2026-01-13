# YOLOv8 目标检测模型训练示例

# 1. 环境准备
# 安装依赖：
# pip install ultralytics opencv-python

import os
import cv2
from ultralytics import YOLO

# 2. 数据准备
# 假设数据结构如下：
# data/
# ├── images/
# │   ├── train/      # 训练图像
# │   └── val/        # 验证图像
# ├── labels/
# │   ├── train/      # 训练标注（YOLO格式）
# │   └── val/        # 验证标注（YOLO格式）
# └── dataset.yaml    # 数据集配置文件

# 创建数据集配置文件
DATASET_CONFIG = """path: data  # 数据集根目录

train: images/train  # 训练图像目录
val: images/val      # 验证图像目录

# 类别名称
names:
  0: person
  1: car
  2: dog
"""

# 3. 训练配置
train_config = {
    "model": "yolov8s.pt",     # 预训练模型（n/s/m/l/x）
    "data": "data/dataset.yaml",  # 数据集配置文件
    "epochs": 100,             # 训练轮数
    "batch": 16,               # 批次大小
    "imgsz": 640,              # 图像大小
    "lr0": 0.01,               # 初始学习率
    "augment": True,           # 数据增强
    "device": "0",            # GPU设备（0表示第一块GPU，"cpu"表示CPU）
    "workers": 4,              # 数据加载线程数
    "save": True,              # 保存模型
    "save_period": 10,         # 每10轮保存一次模型
    "project": "runs/train",  # 训练结果保存目录
    "name": "yolov8s_custom",  # 训练任务名称
    "pretrained": True,        # 使用预训练权重
    "cache": True              # 缓存数据以提高训练速度
}

# 4. 模型训练
def train_model():
    """训练YOLOv8模型"""
    print("开始训练YOLOv8模型...")
    
    # 创建数据集配置文件
    os.makedirs("data", exist_ok=True)
    with open("data/dataset.yaml", "w") as f:
        f.write(DATASET_CONFIG)
    
    # 加载模型
    model = YOLO(train_config["model"])
    
    # 开始训练
    results = model.train(
        data=train_config["data"],
        epochs=train_config["epochs"],
        batch=train_config["batch"],
        imgsz=train_config["imgsz"],
        lr0=train_config["lr0"],
        augment=train_config["augment"],
        device=train_config["device"],
        workers=train_config["workers"],
        save=train_config["save"],
        save_period=train_config["save_period"],
        project=train_config["project"],
        name=train_config["name"],
        pretrained=train_config["pretrained"],
        cache=train_config["cache"]
    )
    
    print("训练完成！")
    print("✅ 模型实际保存路径:", str(results.save_dir))
    print(f"训练结果保存在：{train_config['project']}/{train_config['name']}")
    
    return results

# 5. 模型验证
def validate_model():
    """验证训练好的模型"""
    print("开始验证模型...")
    
    # 加载训练好的模型
    model_path = f"{train_config['project']}/{train_config['name']}/weights/best.pt"
    model = YOLO(model_path)
    
    # 开始验证
    results = model.val(
        data=train_config["data"],
        imgsz=train_config["imgsz"],
        batch=train_config["batch"],
        device=train_config["device"],
        workers=train_config["workers"]
    )
    
    # 打印验证结果
    print("验证结果：")
    print(f"mAP@0.5: {results.box.map50:.4f}")
    # print(f"mAP@0.5:0.95: {results.box.map5095:.4f}")
    print(f"mAP@0.5:0.95: {results.box.map:.4f}")
    
    return results

# 6. 模型推理
def inference_model(image_path="test_image.jpg"):
    """使用训练好的模型进行推理"""
    print(f"开始对图像 {image_path} 进行推理...")
    
    # 加载训练好的模型
    model_path = f"{train_config['project']}/{train_config['name']}/weights/best.pt"
    model = YOLO(model_path)
    
    # 进行推理
    results = model(image_path)
    
    # 可视化结果
    for i, result in enumerate(results):
        # 获取结果图像
        img = result.plot()
        
        # 保存结果图像
        result_image_path = f"result_{i}.jpg"
        cv2.imwrite(result_image_path, img)
        print(f"结果图像已保存到：{result_image_path}")
    
    return results

# 7. 主函数
def main():
    # 训练模型
    train_model()
    
    # 验证模型
    validate_model()
    
    # 推理示例（假设存在test_image.jpg）
    if os.path.exists("test_image.jpg"):
        inference_model("test_image.jpg")
    else:
        print("测试图像不存在，跳过推理示例")

if __name__ == "__main__":
    main()

# 8. 使用说明
# 1. 准备数据集，按照上述数据结构组织
# 2. 修改DATASET_CONFIG中的类别名称和数量
# 3. 调整train_config中的训练参数
# 4. 运行代码：python yolo_train.py
# 5. 训练结果将保存在runs/train/yolov8s_custom目录
# 6. 最佳模型权重为：runs/train/yolov8s_custom/weights/best.pt

# 9. 部署说明
# 可以使用以下方式部署训练好的模型：
# - 命令行推理：yolo task=detect mode=predict model=runs/train/yolov8s_custom/weights/best.pt source=test_image.jpg
# - 导出为ONNX：yolo export model=runs/train/yolov8s_custom/weights/best.pt format=onnx
# - 导出为TensorRT：yolo export model=runs/train/yolov8s_custom/weights/best.pt format=engine