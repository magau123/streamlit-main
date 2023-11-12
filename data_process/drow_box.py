import cv2
import os
import random
import xml.etree.ElementTree as ET

def draw_bounding_boxes(image_folder, annotation_folder, output_dir):
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 遍历图像文件夹中的所有图像文件
    image_files = os.listdir(image_folder)
    for image_file in image_files:
        # 获取图像路径和对应的标注路径
        image_path = os.path.join(image_folder, image_file)
        annotation_file = os.path.splitext(image_file)[0] + '.xml'
        annotation_path = os.path.join(annotation_folder, annotation_file)

        # 检查对应的标注文件是否存在
        if not os.path.isfile(annotation_path):
            continue

        # 读取图像
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        tree = ET.parse(annotation_path)
        root = tree.getroot()

        objects = root.findall('object')

        for obj in objects:
            name = obj.find('name').text
            bbox = obj.find('bndbox')

            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            # 在原图上画框
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # 在框上添加标签
            label = f'{name}'
            cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 保存结果图像
        output_image_path = os.path.join(output_dir, f'{image_file}')
        cv2.imwrite(output_image_path, image)

# 示例用法
image_folder = './output_directory/JPEGImages'
annotation_folder = './output_directory/Annotations'
output_dir = './output_folder'

draw_bounding_boxes(image_folder, annotation_folder, output_dir)
