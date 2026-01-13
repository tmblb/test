#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手动打标签指南

本指南介绍如何使用LabelImg工具进行目标检测数据集的手动标注
"""

import os
import sys


def install_labelimg():
    """
    安装LabelImg标注工具
    """
    print("=== 安装LabelImg标注工具 ===")
    print("方法1：使用pip安装")
    print("pip install labelimg")
    print()
    print("方法2：从源代码安装（推荐）")
    print("git clone https://github.com/tzutalin/labelImg.git")
    print("cd labelImg")
    print("pip install -r requirements.txt")
    print("python labelImg.py")
    print()
    print("方法3：Windows用户可以直接下载exe文件")
    print("https://github.com/tzutalin/labelImg/releases")


def labelimg_usage():
    """
    LabelImg使用指南
    """
    print("=== LabelImg使用指南 ===")
    print("\n1. 启动LabelImg")
    print("   - 命令行启动：labelimg")
    print("   - 或：python labelImg.py")
    
    print("\n2. 设置标注格式")
    print("   - 点击左上角 'Format' 菜单")
    print("   - 选择 'YOLO' 格式（与YOLOv8兼容）")
    print("   - 或选择 'PascalVOC' 格式（通用格式）")
    
    print("\n3. 设置工作目录")
    print("   - 点击左上角 'Open Dir' 选择图像目录")
    print("   - 点击 'Change Save Dir' 选择标注文件保存目录")
    
    print("\n4. 创建类别")
    print("   - 点击左侧 'View' -> 'Create RectBox'")
    print("   - 或使用快捷键：W")
    print("   - 绘制第一个边界框后，会弹出类别输入框")
    print("   - 输入类别名称，如 'person', 'car', 'dog' 等")
    
    print("\n5. 绘制边界框")
    print("   - 使用鼠标左键拖拽绘制边界框")
    print("   - 边界框应准确包围物体")
    print("   - 绘制完成后选择对应的类别")
    
    print("\n6. 编辑边界框")
    print("   - 点击边界框可选中并拖动")
    print("   - 拖拽边界框边角可调整大小")
    print("   - 按 'D' 删除选中的边界框")
    
    print("\n7. 保存标注")
    print("   - 标注完成后，点击 'Save' 按钮")
    print("   - 或使用快捷键：Ctrl + S")
    print("   - YOLO格式会生成 .txt 文件，VOC格式会生成 .xml 文件")
    
    print("\n8. 下一张图像")
    print("   - 点击 'Next Image' 按钮")
    print("   - 或使用快捷键：D")
    print("   - 上一张图像：A")
    
    print("\n9. 快捷键列表")
    print("   - W: 创建边界框")
    print("   - D: 下一张图像")
    print("   - A: 上一张图像")
    print("   - Ctrl + S: 保存")
    print("   - Ctrl + D: 复制当前边界框")
    print("   - Delete: 删除选中的边界框")
    print("   - Ctrl + E: 编辑边界框标签")
    
    print("\n10. 批量处理")
    print("   - 可使用 'Auto Save' 功能自动保存")
    print("   - 'View' -> 'Auto Save Mode'")
    print("   - 标注完成后自动保存并切换到下一张图像")


def labeling_best_practices():
    """
    标注最佳实践
    """
    print("=== 标注最佳实践 ===")
    
    print("\n1. 边界框标注规范")
    print("   - 边界框应准确包围物体，不要包含过多背景")
    print("   - 对于部分遮挡的物体，标注可见部分")
    print("   - 边界框应与物体边缘保持适当距离（1-2像素）")
    print("   - 对于重叠物体，分别标注各自的可见部分")
    
    print("\n2. 类别标注规范")
    print("   - 使用清晰、简洁的类别名称")
    print("   - 保持类别名称的一致性")
    print("   - 避免使用模糊或歧义的类别名称")
    print("   - 对于不确定的物体，可标注为 'unknown' 或暂时跳过")
    
    print("\n3. 标注质量控制")
    print("   - 标注完成后，抽样检查标注质量")
    print("   - 检查是否有漏标、误标或重复标注")
    print("   - 确保标注文件与图像文件一一对应")
    print("   - 使用工具（如 labelImg 的 'Verify Image' 功能）检查标注完整性")
    
    print("\n4. 标注效率提升")
    print("   - 使用快捷键提高标注速度")
    print("   - 对于相似物体，使用复制功能（Ctrl + D）")
    print("   - 使用批量处理功能自动保存和切换图像")
    print("   - 合理安排标注时间，避免疲劳导致的标注质量下降")
    
    print("\n5. 特殊情况处理")
    print("   - 对于小物体：适当放大图像后标注")
    print("   - 对于模糊物体：根据上下文判断类别")
    print("   - 对于复杂场景：优先标注主要物体")
    
    print("\n6. 标注文件管理")
    print("   - 保持标注文件与图像文件同名")
    print("   - 使用统一的目录结构存放标注文件")
    print("   - 定期备份标注文件")
    print("   - 使用版本控制工具（如Git）管理标注文件")


def yolo_format_explanation():
    """
    YOLO标注格式说明
    """
    print("=== YOLO标注格式说明 ===")
    print("\nYOLO格式的标注文件是一个.txt文件，与对应的图像文件同名")
    print("每个物体占用一行，格式如下：")
    print("<class_id> <x_center> <y_center> <width> <height>")
    
    print("\n各字段含义：")
    print("- <class_id>: 类别ID（从0开始的整数）")
    print("- <x_center>: 边界框中心x坐标（归一化到0-1之间）")
    print("- <y_center>: 边界框中心y坐标（归一化到0-1之间）")
    print("- <width>: 边界框宽度（归一化到0-1之间）")
    print("- <height>: 边界框高度（归一化到0-1之间）")
    
    print("\n示例：")
    print("0 0.45 0.35 0.20 0.30")
    print("1 0.70 0.60 0.30 0.40")
    
    print("\n类别映射：")
    print("需要一个单独的配置文件（如dataset.yaml）来定义类别ID与名称的映射")
    print("示例：")
    print("names:")
    print("  0: person")
    print("  1: car")
    print("  2: dog")


def convert_format():
    """
    标注格式转换
    """
    print("=== 标注格式转换 ===")
    print("\n1. VOC格式转YOLO格式")
    print("可以使用以下工具：")
    print("- LabelImg自带转换功能")
    print("- Roboflow在线转换工具：https://roboflow.com/")
    print("- 自定义转换脚本")
    
    print("\n2. YOLO格式转VOC格式")
    print("可以使用：")
    print("- LabelImg（打开YOLO格式标注，重新保存为VOC格式）")
    print("- 自定义转换脚本")
    
    print("\n3. 批量转换工具")
    print("- Ultralytics YOLO提供的转换工具")
    print("- CVAT标注平台的导出功能")
    print("- LabelStudio的格式转换功能")


def main():
    """
    主函数
    """
    print("=====================================")
    print("         手动打标签指南")
    print("=====================================")
    print()
    
    print("本指南介绍如何使用LabelImg工具进行目标检测数据集的手动标注")
    print()
    
    while True:
        print("请选择需要查看的内容：")
        print("1. 安装LabelImg标注工具")
        print("2. LabelImg使用指南")
        print("3. 标注最佳实践")
        print("4. YOLO标注格式说明")
        print("5. 标注格式转换")
        print("0. 退出")
        print()
        
        choice = input("请输入选项编号：")
        print()
        
        if choice == "1":
            install_labelimg()
        elif choice == "2":
            labelimg_usage()
        elif choice == "3":
            labeling_best_practices()
        elif choice == "4":
            yolo_format_explanation()
        elif choice == "5":
            convert_format()
        elif choice == "0":
            print("感谢使用手动打标签指南！")
            break
        else:
            print("无效选项，请重新输入！")
        
        print()
        input("按Enter键继续...")
        print()


if __name__ == "__main__":
    main()
