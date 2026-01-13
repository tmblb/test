# 目标检测模型训练指南

## 一、项目结构

```
.
├── dataset_preparation.py  # 数据集准备工具
├── yolo_train.py           # YOLOv8模型训练代码
├── image_recognition.py    # 简单图像识别示例
└── test.py                 # 测试文件
```

## 二、数据集准备

### 1. 安装依赖

#### 方法1：使用requirements.txt文件
```bash
pip install -r requirements.txt
```

#### 方法2：手动安装核心依赖
```bash
pip install ultralytics opencv-python requests tqdm numpy argparse
```

### 2. 使用数据集准备工具

数据集准备工具可以自动完成以下任务：
- 创建数据集目录结构
- 下载COCO2017样本图像
- 使用预训练YOLOv8模型自动标注图像
- 划分数据集（训练集/验证集/测试集）
- 生成YOLO格式的数据集配置文件

#### 基本用法

```bash
python dataset_preparation.py
```

#### 参数说明

```bash
python dataset_preparation.py --help
usage: dataset_preparation.py [-h] [--root_dir ROOT_DIR] [--num_images NUM_IMAGES] [--confidence CONFIDENCE] [--train_ratio TRAIN_RATIO] [--val_ratio VAL_RATIO] [--test_ratio TEST_RATIO]

数据集准备工具

optional arguments:
  -h, --help            show this help message and exit
  --root_dir ROOT_DIR   数据集根目录
  --num_images NUM_IMAGES
                        下载图像数量
  --confidence CONFIDENCE
                        自动标注的置信度阈值
  --train_ratio TRAIN_RATIO
                        训练集比例
  --val_ratio VAL_RATIO
                        验证集比例
  --test_ratio TEST_RATIO
                        测试集比例
```

#### 示例：自定义参数

```bash
python dataset_preparation.py \
    --root_dir my_dataset \
    --num_images 100 \
    --confidence 0.7 \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1
```

### 3. 数据集目录结构

工具会自动创建以下目录结构：

```
data/
├── images/
│   ├── train/      # 训练图像
│   ├── val/        # 验证图像
│   └── test/       # 测试图像
├── labels/
│   ├── train/      # 训练标注（YOLO格式）
│   ├── val/        # 验证标注（YOLO格式）
│   └── test/       # 测试标注（YOLO格式）
└── dataset.yaml    # 数据集配置文件
```

## 三、模型训练

### 1. 使用YOLOv8训练代码

```bash
python yolo_train.py
```

### 2. 训练配置说明

在`yolo_train.py`中，你可以修改以下训练配置：

```python
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
```

### 3. 训练结果

训练完成后，结果将保存在`runs/train/yolov8s_custom`目录中：

```
runs/train/yolov8s_custom/
├── weights/
│   ├── best.pt    # 最佳模型权重
│   └── last.pt    # 最后一轮模型权重
├── results.csv    # 训练结果记录
├── confusion_matrix.png  # 混淆矩阵
├── PR_curve.png   # PR曲线
└── val_batch0_pred.jpg   # 验证集预测结果示例
```

## 四、模型验证和推理

### 1. 模型验证

```bash
yolo val model=runs/train/yolov8s_custom/weights/best.pt data=data/dataset.yaml
```

### 2. 模型推理

```bash
yolo predict model=runs/train/yolov8s_custom/weights/best.pt source=test_image.jpg
```

## 五、模型部署

### 1. 导出为ONNX格式

```bash
yolo export model=runs/train/yolov8s_custom/weights/best.pt format=onnx
```

### 2. 导出为TensorRT格式

```bash
yolo export model=runs/train/yolov8s_custom/weights/best.pt format=engine
```

### 3. 导出为其他格式

```bash
# CoreML格式
yolo export model=runs/train/yolov8s_custom/weights/best.pt format=coreml

# TensorFlow SavedModel格式
yolo export model=runs/train/yolov8s_custom/weights/best.pt format=saved_model

# TensorFlow Lite格式
yolo export model=runs/train/yolov8s_custom/weights/best.pt format=tflite
```

## 六、注意事项

1. **数据质量**：确保图像清晰，标注准确
2. **数据量**：对于简单任务，建议至少100-200张图像；对于复杂任务，建议1000张以上
3. **类别平衡**：避免某一类别的数据过多或过少
4. **硬件要求**：
   - CPU：可以训练，但速度较慢
   - GPU：推荐使用NVIDIA GPU，至少4GB显存
5. **内存要求**：建议至少8GB内存

## 七、常见问题

### 1. 模型训练时出现内存不足

解决方案：
- 减小批次大小（batch参数）
- 减小图像大小（imgsz参数）
- 使用更小的模型（如从yolov8s换成yolov8n）

### 2. 模型精度低

解决方案：
- 增加训练轮数
- 提高数据质量和数量
- 调整学习率
- 增加数据增强

### 3. 模型过拟合

解决方案：
- 增加数据量
- 加强数据增强
- 降低模型复杂度
- 增加正则化

### 4. 模型欠拟合

解决方案：
- 增加模型复杂度
- 延长训练时间
- 提高学习率
- 减少正则化

## 八、进阶技巧

1. **超参数调优**：使用Optuna、Hyperopt等工具自动调参
2. **迁移学习**：使用领域相关的预训练模型
3. **数据增强**：自定义数据增强策略
4. **模型融合**：结合不同大小或训练策略的模型
5. **后处理优化**：调整非极大值抑制参数、置信度阈值等

## 九、参考资料

1. [Ultralytics YOLOv8官方文档](https://docs.ultralytics.com/)
2. [COCO数据集官方网站](https://cocodataset.org/)
3. [OpenCV官方文档](https://opencv.org/)
4. [PyTorch官方文档](https://pytorch.org/)
