import cv2
import os
import random
import xml.etree.ElementTree as ET

def crop_augmentation(image_folder, annotation_folder, output_dir, crop_size, num_augmentations):
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'JPEGImages'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'Annotations'), exist_ok=True)

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

        # 在原图上进行裁剪增强
        for i in range(num_augmentations):
            # 随机选择裁剪框的位置
            x = random.randint(0, width - crop_size[1])
            y = random.randint(0, height - crop_size[0])

            # 裁剪图像
            cropped_image = image[y:y+crop_size[0], x:x+crop_size[1]]

            # 调整对应的标注框
            augmented_objects = []
            for obj in objects:
                name = obj.find('name').text
                bbox = obj.find('bndbox')

                xmin = max(int(bbox.find('xmin').text) - x, 0)
                ymin = max(int(bbox.find('ymin').text) - y, 0)
                xmax = min(int(bbox.find('xmax').text) - x, crop_size[1])
                ymax = min(int(bbox.find('ymax').text) - y, crop_size[0])

                # 忽略裁剪后过小的标注框和超出裁剪框范围的标注框
                if xmax > 0 and ymax > 0 and xmin < crop_size[1] and ymin < crop_size[0]:
                    new_bbox = {
                        'xmin': xmin,
                        'ymin': ymin,
                        'xmax': xmax,
                        'ymax': ymax
                    }

                    augmented_object = {
                        'name': name,
                        'bbox': new_bbox
                    }

                    augmented_objects.append(augmented_object)

            # 保存增强后的图像
            output_image_path = os.path.join(output_dir, 'JPEGImages', f'{os.path.splitext(image_file)[0]}_augmented_{i}.jpg')
            cv2.imwrite(output_image_path, cropped_image)

            # 保存增强后的标注
            output_annotation_path = os.path.join(output_dir, 'Annotations', f'{os.path.splitext(annotation_file)[0]}_augmented_{i}.xml')
            create_voc_annotation(output_annotation_path, augmented_objects, crop_size)

def create_voc_annotation(output_path, objects, crop_size):
    root = ET.Element('annotation')

    folder_elem = ET.SubElement(root, 'folder')
    folder_elem.text = 'VOC2007'

    filename_elem = ET.SubElement(root, 'filename')
    filename_elem.text = os.path.basename(output_path)  # Use the filename of the output annotation

    size_elem = ET.SubElement(root, 'size')
    width_elem = ET.SubElement(size_elem, 'width')
    height_elem = ET.SubElement(size_elem, 'height')
    depth_elem = ET.SubElement(size_elem, 'depth')

    height_elem.text = str(crop_size[0])
    width_elem.text = str(crop_size[1])
    depth_elem.text = '3'  # Assuming 3 channels (RGB)

    for obj in objects:
        object_elem = ET.SubElement(root, 'object')
        name_elem = ET.SubElement(object_elem, 'name')
        bndbox_elem = ET.SubElement(object_elem, 'bndbox')

        name_elem.text = obj['name']

        xmin_elem = ET.SubElement(bndbox_elem, 'xmin')
        ymin_elem = ET.SubElement(bndbox_elem, 'ymin')
        xmax_elem = ET.SubElement(bndbox_elem, 'xmax')
        ymax_elem = ET.SubElement(bndbox_elem, 'ymax')

        xmin_elem.text = str(obj['bbox']['xmin'])
        ymin_elem.text = str(obj['bbox']['ymin'])
        xmax_elem.text = str(obj['bbox']['xmax'])
        ymax_elem.text = str(obj['bbox']['ymax'])

    tree = ET.ElementTree(root)
    tree.write(output_path)



# 示例用法
image_folder = './input_data'
annotation_folder = './input_label'
output_dir = './output_directory'
crop_size = (380, 380)  # 裁剪框的大小为 200x200
num_augmentations = 10

crop_augmentation(image_folder, annotation_folder, output_dir, crop_size, num_augmentations)
