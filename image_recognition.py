import cv2
import requests
import numpy as np
import os

def download_image(url, save_path):
    """下载图像到本地"""
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)
    print(f"图像已下载到: {save_path}")

def detect_faces(image_path):
    """使用OpenCV的Haar级联分类器检测人脸"""
    # 加载预训练的人脸检测模型
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 读取图像
    img = cv2.imread(image_path)
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # 在图像上绘制人脸边界框
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # 保存结果图像
    result_path = 'detected_faces.jpg'
    cv2.imwrite(result_path, img)
    
    print(f"检测到 {len(faces)} 个人脸")
    print(f"结果图像已保存到: {result_path}")
    
    return len(faces)

def main():
    # 示例图像URL
    # image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Albert_Einstein_Head.jpg/330px-Albert_Einstein_Head.jpg"
    image_url = 'https://cn.bing.com/images/search?view=detailV2&ccid=7awM56%2BQ&id=5EB56591A606E7F5AFF23FAF881659D3E24B2C46&thid=OIP.7awM56-QA5ofz4xyntSD-wHaE9&mediaurl=https%3A%2F%2Fgd-hbimg.huaban.com%2F26eb683dc4d4cbca5092d6366d12f2ea99a6d24725902-kFbH2T_fw658&exph=441&expw=658&q=%E4%BA%BA&FORM=IRPRST&ck=8B1619531F07E46281F424426DBB35A3&selectedIndex=19&itb=0&cw=1528&ch=740&ajaxhist=0&ajaxserp=0'
    
    # 下载图像
    image_path = 'test_image.jpg'
    download_image(image_url, image_path)
    
    # 检测人脸
    face_count = detect_faces(image_path)
    
    # 清理临时文件
    os.remove(image_path)
    
    print("\n图像识别完成！")

if __name__ == "__main__":
    main()