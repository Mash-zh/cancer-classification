import h5py
import numpy as np
import matplotlib.pyplot as plt
import os


def h5_to_images(h5_file_path, output_folder, dataset_name="images"):
    """
    将 .h5 文件中的数据保存为图片。

    Args:
        h5_file_path (str): .h5 文件路径。
        output_folder (str): 输出图片的文件夹路径。
        dataset_name (str): 数据集名称，默认是 "images"。
    """
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开 .h5 文件
    with h5py.File(h5_file_path, 'r') as h5_file:
        # 获取数据集
        if dataset_name not in h5_file:
            print(f"Dataset '{dataset_name}' not found in {h5_file_path}.")
            return

        data = h5_file[dataset_name][:]

        # 确保数据是图像格式 (例如，N x H x W 或 N x H x W x C)
        if len(data.shape) < 3 or len(data.shape) > 4:
            print(f"Unexpected data shape: {data.shape}. Expected 3D or 4D array.")
            return

        # 保存每一张图片
        for i, img in enumerate(data):
            if len(img.shape) == 3 and img.shape[2] == 1:  # 单通道图像
                img = img.squeeze(-1)  # 去掉通道维度
            elif len(img.shape) == 3 and img.shape[2] > 3:  # 非图像格式
                print(f"Skipping image {i} with shape {img.shape}.")
                continue
            if i < 10:
                prefix = '0000'
            elif i < 100:
                prefix = '000'
            elif i < 1000:
                prefix = '00'
            elif i < 10000:
                prefix = '0'
            else:
                prefix = ''
            plt.imsave(os.path.join(output_folder, f"{prefix}{i}.png"), img, cmap='gray' if len(img.shape) == 2 else None)
            print(f"Saved {i}.png")

    print(f"Images saved to {output_folder}")

if __name__ == '__main__':
    # 使用示例
    h5_file_path = "data/cancer/test_x.h5"  # 替换为你的 .h5 文件路径
    output_folder = "data/cancer/test_all"
    dataset_name = "x"  # 替换为你的数据集名称
    h5_to_images(h5_file_path, output_folder, dataset_name)
