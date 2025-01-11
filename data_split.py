import os
import shutil


def prepare_celeba_dataset(src_images_dir, list_file, target_dir):
    """
    根据 CelebA 数据集的分割文件将图像分配到训练集、验证集和测试集文件夹中。

    参数：
    - src_images_dir: 原始图像所在目录
    - list_file: 数据集划分的文件，指示图像属于训练集、验证集或测试集
    - target_dir: 存储目标文件夹路径（例如train、val、test）
    """
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)

    # 读取 list_landmarks_celeba.txt 文件
    with open(list_file, 'r') as f:
        lines = f.readlines()

    # 跳过第一行标题
    lines = lines[2:]

    for line in lines:
        img_name, set_type = line.split()
        set_type = int(set_type.strip())

        # 将扩展名从 .jpg 替换为 .png
        img_name = img_name.replace('.jpg', '.png')

        if set_type == 0:  # 训练集
            target_subdir = os.path.join(target_dir, 'train')
        elif set_type == 1:  # 验证集
            target_subdir = os.path.join(target_dir, 'val')
        elif set_type == 2:  # 测试集
            target_subdir = os.path.join(target_dir, 'test')
        else:
            continue

        # 确保目标目录存在
        os.makedirs(target_subdir, exist_ok=True)

        # 移动文件到对应目录
        src_image_path = os.path.join(src_images_dir, img_name)
        target_image_path = os.path.join(target_subdir, img_name)

        if os.path.exists(src_image_path):
            shutil.copy(src_image_path, target_image_path)
        else:
            print(f"Image {img_name} not found in source directory!")


# 使用方法示例
src_images_dir = 'edge-connect-master/CelebA/Img/img_align_celeba_png.7z/img_align_celeba_png'  # .png形式的路径
list_file = 'edge-connect-master/CelebA/Eval/list_eval_partition.txt'  # 根据 Eval 文件夹中的该 txt 文件
target_dir = 'edge-connect-master/CelebA/split_images'  # 目标数据集目录
prepare_celeba_dataset(src_images_dir, list_file, target_dir)
