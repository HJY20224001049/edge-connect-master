import cv2
import numpy as np
import random
import os
print("当前工作目录为:", os.getcwd())
# 使用原始字符串表示文件路径，避免转义字符问题
image_path = r'D:\project\python\dl_project\edge-connect-master\images\hat.png'
image = cv2.imread(image_path)
if image is None:
    print(f"无法读取图像 {image_path}，请检查文件是否存在及格式是否正确")
    raise SystemExit(1)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 创建一个全肤色的遮罩 (浅肤色)
mask = np.zeros_like(image)

# 设置肤色 (浅肤色)
skin_color = (255, 255, 255)  # RGB值，表示浅肤色

# 定义遮罩的形状和大小相关参数
max_shape_size = 0.2  # 最大形状大小
min_shape_size = 0.1  # 最小形状大小
max_area_fraction = 0.05  # 遮罩最大面积占图像面积的比例

# 计算允许的最大遮罩面积
max_mask_area = gray_image.size * max_area_fraction

# 生成随机线条或多边形并添加到遮罩中
current_area = 0
while current_area < max_mask_area:
    shape_type = random.choice(['line', 'polygon'])

    if shape_type == 'line':
        # 随机生成一条直线
        x1 = random.randint(0, mask.shape[1])
        y1 = random.randint(0, mask.shape[0])
        x2 = random.randint(0, mask.shape[1])
        y2 = random.randint(0, mask.shape[0])
        thickness = random.randint(1, 3)  # 线条厚度
        cv2.line(mask, (x1, y1), (x2, y2), skin_color, thickness)

    elif shape_type == 'polygon':
        # 随机生成一个多边形
        num_points = random.randint(3, 6)  # 多边形的顶点数量
        points = []
        for _ in range(num_points):
            points.append([random.randint(0, mask.shape[1]), random.randint(0, mask.shape[0])])
        points = np.array(points, np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(mask, [points], isClosed=True, color=skin_color, thickness=2)

    # 计算当前遮罩的面积
    current_area = np.sum(np.all(mask == skin_color, axis=-1))

# 将遮罩图像保存为文件，例如保存为'mask.png'，这里文件名可根据需求自行设定
cv2.imwrite('hat1.png', mask)