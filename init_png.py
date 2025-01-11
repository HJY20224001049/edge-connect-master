import cv2
import numpy as np


def combine_images(image_path1, image_path2, result_path):
    """
    函数功能：读取两张图片并进行合成，然后将合成结果保存到指定路径

    参数：
    image_path1：第一张图片的完整路径
    image_path2：第二张图片的完整路径
    result_path：合成结果图片保存的完整路径
    """
    # 读取第一张图片
    a = cv2.imread(image_path1)
    if a is None:
        print(f'无法读取图像 {image_path1}，请检查文件是否存在及格式是否正确')
        return
    # 读取第二张图片
    b = cv2.imread(image_path2)
    if b is None:
        print(f'无法读取图像 {image_path2}，请检查文件是否存在及格式是否正确')
        return
    # 调整两张图片尺寸为相同大小（这里统一调整为256x256，可根据需求修改尺寸）
    a = cv2.resize(a, (256, 256))
    b = cv2.resize(b, (256, 256))
    # 使用cv2.add函数进行图像合成，也可根据需求换用其他合成方式，如cv2.addWeighted等
    combined_image = cv2.add(a, b)
    # 将合成后的图像保存到指定路径
    cv2.imwrite(result_path, combined_image)
    print(f'图片已成功合成并保存至 {result_path}')


if __name__ == '__main__':
    # 这里替换为你实际的两张图片的路径以及合成结果要保存的路径
    image_path1 = r"D:\project\python\dl_project\edge-connect-master\dude.png"
    image_path2 = r"D:\project\python\dl_project\edge-connect-master\data\masks\dude_masked.png"
    result_path = r"D:\project\python\dl_project\edge-connect-master\data\images\dude_combined.png"
    combine_images(image_path1, image_path2, result_path)

