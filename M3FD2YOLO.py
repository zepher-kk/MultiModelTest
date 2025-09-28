import os
import xml.etree.ElementTree as ET
import random
from shutil import copyfile
from tqdm import tqdm

# M3FD数据集类别定义
classes = ["People", "Car", "Bus", "Motorcycle", "Lamp", "Truck"]


def xyxy2xywh(size, box):
    """将VOC格式的边界框转换为YOLO格式"""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[2]) / 2.0 * dw
    y = (box[1] + box[3]) / 2.0 * dh
    w = (box[2] - box[0]) * dw
    h = (box[3] - box[1]) * dh
    return (x, y, w, h)


def convert_m3fd_to_yolo_with_split(source_dir, target_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                                    sample_ratio=0.1):
    """
    将未划分的M3FD数据集转换为YOLO格式并划分train/val/test

    Args:
        source_dir: 原始M3FD目录路径（包含Annotation, Ir, Vis文件夹）
        target_dir: 目标YOLO格式目录路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        sample_ratio: 采样比例（仅处理数据集的百分比）
    """

    # 设置随机种子以确保可重复性
    random.seed(42)

    # 源目录路径
    annotation_dir = os.path.join(source_dir, "Annotation")
    ir_dir = os.path.join(source_dir, "Ir")
    vis_dir = os.path.join(source_dir, "Vis")

    # 检查目录是否存在
    for dir_path in [annotation_dir, ir_dir, vis_dir]:
        if not os.path.exists(dir_path):
            print(f"错误: 目录不存在 - {dir_path}")
            return

    # 获取所有XML文件（不包含扩展名）
    xml_files = [f for f in os.listdir(annotation_dir) if f.endswith('.xml')]
    base_names = [os.path.splitext(f)[0] for f in xml_files]

    print(f"找到 {len(base_names)} 个样本")

    # 仅采样10%的数据
    n_total = len(base_names)
    n_sample = int(n_total * sample_ratio)
    base_names = random.sample(base_names, n_sample)
    print(f"采样 {len(base_names)} 个样本 ({sample_ratio * 100}%)")

    # 随机打乱并划分数据集
    random.shuffle(base_names)
    n_total = len(base_names)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_set = base_names[:n_train]
    val_set = base_names[n_train:n_train + n_val]
    test_set = base_names[n_train + n_val:]

    print(f"数据集划分: 训练集 {n_train}, 验证集 {n_val}, 测试集 {n_test}")

    # 创建目标目录结构
    splits = ['train', 'val', 'test']
    modalities = ['images', 'images_ir']  # 可见光和红外

    for split in splits:
        for modality in modalities:
            img_dir = os.path.join(target_dir, modality, split)
            os.makedirs(img_dir, exist_ok=True)

        label_dir = os.path.join(target_dir, 'labels', split)
        os.makedirs(label_dir, exist_ok=True)

    # 处理每个划分
    split_data = {
        'train': train_set,
        'val': val_set,
        'test': test_set
    }

    for split, samples in split_data.items():
        print(f"\n处理 {split} 划分 ({len(samples)} 个样本)...")

        for base_name in tqdm(samples):
            # 处理可见光模态
            vis_source_path = None
            for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                candidate = os.path.join(vis_dir, base_name + ext)
                if os.path.exists(candidate):
                    vis_source_path = candidate
                    break

            if vis_source_path:
                vis_target_path = os.path.join(target_dir, 'images', split, os.path.basename(vis_source_path))
                copyfile(vis_source_path, vis_target_path)

            # 处理红外模态
            ir_source_path = None
            for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                candidate = os.path.join(ir_dir, base_name + ext)
                if os.path.exists(candidate):
                    ir_source_path = candidate
                    break

            if ir_source_path:
                ir_target_path = os.path.join(target_dir, 'images_ir', split, os.path.basename(ir_source_path))
                copyfile(ir_source_path, ir_target_path)

            # 转换标签文件
            xml_path = os.path.join(annotation_dir, base_name + '.xml')
            txt_path = os.path.join(target_dir, 'labels', split, base_name + '.txt')

            if os.path.exists(xml_path):
                convert_xml_to_yolo(xml_path, txt_path)

    # 创建data.yaml配置文件
    create_data_yaml(target_dir, classes)

    print(f"\n转换完成！数据集已保存到: {target_dir}")


def convert_xml_to_yolo(xml_path, txt_path):
    """将单个XML文件转换为YOLO格式的TXT文件"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 获取图像尺寸
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)

        # 收集所有对象的标注
        annotations = []
        for obj in root.findall('object'):
            # 获取类别名称
            cls_name = obj.find('name').text

            # 跳过不在类别列表中的对象
            if cls_name not in classes:
                continue

            # 获取类别ID
            cls_id = classes.index(cls_name)

            # 获取边界框坐标
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            # 转换为YOLO格式
            yolo_bbox = xyxy2xywh((img_width, img_height), (xmin, ymin, xmax, ymax))

            # 添加到标注列表
            annotations.append(f"{cls_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}")

        # 写入YOLO格式的TXT文件
        if annotations:
            with open(txt_path, 'w') as f:
                f.write('\n'.join(annotations))
            return True
        return False

    except Exception as e:
        print(f"处理 {xml_path} 时出错: {e}")
        return False


def create_data_yaml(target_dir, classes):
    """创建YOLO格式的data.yaml配置文件"""
    yaml_content = f"""# M3FD数据集 - YOLO格式
# 多模态数据集（可见光和红外）

path: {target_dir}

# 图像路径
train: images/train
val: images/val
test: images/test

# 红外图像路径（可选）
ir_train: images_ir/train
ir_val: images_ir/val
ir_test: images_ir/test

# 标签路径
train_labels: labels/train
val_labels: labels/val
test_labels: labels/test

# 类别信息
nc: {len(classes)}
names: {classes}

# 数据集信息
description: M3FD多模态数据集（可见光+红外）
modalities: [visible, infrared]
"""

    yaml_path = os.path.join(target_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"创建数据集配置文件: {yaml_path}")


if __name__ == '__main__':
    # 配置路径
    SOURCE_DIR = "D:/learn/dmt/MM/M3FD"  # 替换为您的M3FD目录路径
    TARGET_DIR = "D:/learn/dmt/MM/M3FD_yolo_10percent"  # 替换为目标YOLO格式目录路径

    # 执行转换 - 只处理10%的数据
    convert_m3fd_to_yolo_with_split(
        SOURCE_DIR,
        TARGET_DIR,
        sample_ratio=0.1  # 只处理10%的数据
    )