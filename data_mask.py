import os
import cv2
import numpy as np
import random


def generate_irregular_mask(image, num_vertices=3, brush_strokes=1, min_thickness=3, max_thickness=10):
    """
    生成不规则形状的遮罩图像，进一步减少遮罩的覆盖面积。遮罩区域为白色（255），未遮挡区域为黑色（0）。

    参数：
    - image: 输入的图像
    - num_vertices: 每个不规则多边形的顶点数，较少的顶点数生成的多边形较小
    - brush_strokes: 使用画笔次数，较少的笔触次数会减少遮挡区域
    - min_thickness: 最小线条厚度，较小的厚度会减少覆盖面积
    - max_thickness: 最大线条厚度，较小的厚度会减少覆盖面积

    返回值：
    - mask: 生成的遮罩图像
    """
    height, width, _ = image.shape
    mask = np.zeros((height, width), dtype=np.uint8)  # 全黑遮罩

    # 随机生成多边形遮挡
    for _ in range(brush_strokes):
        num_points = random.randint(3, num_vertices)  # 顶点数减少
        points = []
        for _ in range(num_points):
            x = random.randint(0, width)
            y = random.randint(0, height)
            points.append([x, y])

        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)

    # 使用画笔模拟效果（随机生成路径）
    for _ in range(brush_strokes):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)
        thickness = random.randint(min_thickness, max_thickness)  # 控制线条厚度范围
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)

    return mask


def prepare_masks(src_images_dir, target_mask_dir):
    """
    根据 CelebA 数据集中的图像生成不规则遮罩并保存。

    参数：
    - src_images_dir: 输入图像所在目录
    - target_mask_dir: 存储遮罩图像的目录
    """
    # 确保目标遮罩目录存在
    os.makedirs(target_mask_dir, exist_ok=True)

    # 获取所有图片文件
    image_files = [f for f in os.listdir(src_images_dir) if f.endswith('.png') or f.endswith('.jpg')]

    for img_file in image_files:
        # 读取图像
        img_path = os.path.join(src_images_dir, img_file)
        image = cv2.imread(img_path)

        if image is None:
            print(f"Failed to read image {img_file}")
            continue

        # 生成不规则遮罩，进一步减少遮罩覆盖面积
        mask = generate_irregular_mask(image)

        # 保存遮罩图像
        mask_path = os.path.join(target_mask_dir, img_file)
        success = cv2.imwrite(mask_path, mask)
        if success:
            print(f"Mask successfully saved for {img_file}")
        else:
            print(f"Failed to save mask for {img_file}")


# 使用方法示例
src_images_dir = 'E:/college/third_grade/9-12/deeplearning/project/edge-connect-master/edge-connect-master/CelebA/split_images/test'  # 要生成遮罩图片的文件夹
target_mask_dir = 'E:/college/third_grade/9-12/deeplearning/project/edge-connect-master/edge-connect-master/CelebA/mask_split_images/mask_test' # 输出图片的目标文件夹
prepare_masks(src_images_dir, target_mask_dir)
