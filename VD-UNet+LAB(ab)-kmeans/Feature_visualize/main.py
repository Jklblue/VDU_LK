import torch
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
def visualize_feature_map(img_batch,out_path,type,BI):
    feature_map = torch.squeeze(img_batch)
    feature_map = feature_map.detach().cpu().numpy()

    feature_map_sum = feature_map[0, :, :]
    feature_map_sum = np.expand_dims(feature_map_sum, axis=2)
    for i in range(0, 2048):
        feature_map_split = feature_map[i,:, :]
        feature_map_split = np.expand_dims(feature_map_split,axis=2)
        if i > 0:
            feature_map_sum +=feature_map_split
        feature_map_split = BI.transform(feature_map_split)

        plt.imshow(feature_map_split)
        plt.savefig(out_path + str(i) + "_{}.jpg".format(type) )
        plt.xticks()
        plt.yticks()
        plt.axis('off')

    feature_map_sum = BI.transform(feature_map_sum)
    plt.imshow(feature_map_sum)
    plt.savefig(out_path + "sum_{}.jpg".format(type))
    print("save sum_{}.jpg".format(type))



class BilinearInterpolation(object):
    def __init__(self, w_rate: float, h_rate: float, *, align='center'):
        if align not in ['center', 'left']:
            logging.exception(f'{align} is not a valid align parameter')
            align = 'center'
        self.align = align
        self.w_rate = w_rate
        self.h_rate = h_rate

    def set_rate(self,w_rate: float, h_rate: float):
        self.w_rate = w_rate    # w 的缩放率
        self.h_rate = h_rate    # h 的缩放率

    # 由变换后的像素坐标得到原图像的坐标    针对高
    def get_src_h(self, dst_i,source_h,goal_h) -> float:
        if self.align == 'left':
            # 左上角对齐
            src_i = float(dst_i * (source_h/goal_h))
        elif self.align == 'center':
            # 将两个图像的几何中心重合。
            src_i = float((dst_i + 0.5) * (source_h/goal_h) - 0.5)
        src_i += 0.001
        src_i = max(0.0, src_i)
        src_i = min(float(source_h - 1), src_i)
        return src_i
    # 由变换后的像素坐标得到原图像的坐标    针对宽
    def get_src_w(self, dst_j,source_w,goal_w) -> float:
        if self.align == 'left':
            # 左上角对齐
            src_j = float(dst_j * (source_w/goal_w))
        elif self.align == 'center':
            # 将两个图像的几何中心重合。
            src_j = float((dst_j + 0.5) * (source_w/goal_w) - 0.5)
        src_j += 0.001
        src_j = max(0.0, src_j)
        src_j = min((source_w - 1), src_j)
        return src_j

    def transform(self, img):
        source_h, source_w, source_c = img.shape  # (235, 234, 3)
        goal_h, goal_w = round(
            source_h * self.h_rate), round(source_w * self.w_rate)
        new_img = np.zeros((goal_h, goal_w, source_c), dtype=np.uint8)

        for i in range(new_img.shape[0]):       # h
            src_i = self.get_src_h(i,source_h,goal_h)
            for j in range(new_img.shape[1]):
                src_j = self.get_src_w(j,source_w,goal_w)
                i2 = ceil(src_i)
                i1 = int(src_i)
                j2 = ceil(src_j)
                j1 = int(src_j)
                x2_x = j2 - src_j
                x_x1 = src_j - j1
                y2_y = i2 - src_i
                y_y1 = src_i - i1
                new_img[i, j] = img[i1, j1]*x2_x*y2_y + img[i1, j2] * \
                    x_x1*y2_y + img[i2, j1]*x2_x*y_y1 + img[i2, j2]*x_x1*y_y1
        return new_img
#使用方法

import matplotlib.pyplot as plt
import torch
import torchvision
import math


def find_closest_factors(n):
    """
    找到最接近的两个因数作为子图的行数和列数
    :param n: 通道数
    :return: 行数和列数
    """
    root = int(math.sqrt(n))
    while n % root != 0:
        root -= 1
    return root, n // root


def visualize_channels_separately(feature_map, num_channels_to_show):
    """
    分别可视化特征图的每个通道
    :param feature_map: 输入的特征图张量，形状为 (1, C, H, W)
    :param num_channels_to_show: 要展示的通道数
    """
    # 去掉批量维度
    feature_map = feature_map.squeeze(0)

    # 找到最接近的两个因数作为子图的行数和列数
    rows, cols = find_closest_factors(num_channels_to_show)

    # 创建子图布局
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))

    # 如果只有一个子图，axes 不是二维数组，需要特殊处理
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]

    # 分别可视化每个通道的特征图
    for i in range(num_channels_to_show):
        row = i // cols
        col = i % cols
        axes[row][col].imshow(feature_map[i].cpu().numpy(), cmap='gray')
        axes[row][col].set_title(f'Channel {i + 1}')
        axes[row][col].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_channels_as_grid(feature_map, num_channels_to_show):
    """
    将特征图的通道拼接成网格图进行可视化
    :param feature_map: 输入的特征图张量，形状为 (1, C, H, W)
    :param num_channels_to_show: 要展示的通道数
    """
    # 去掉批量维度
    feature_map = feature_map.squeeze(0)

    # 只选取要展示的通道
    feature_map = feature_map[:num_channels_to_show]

    # 使用 make_grid 函数将特征图拼接成网格图
    feature_map_grid = torchvision.utils.make_grid(feature_map.unsqueeze(1))

    # 将张量从 GPU 移动到 CPU，并转换为 NumPy 数组，同时调整维度
    feature_map_grid_np = feature_map_grid.cpu().permute(1, 2, 0).numpy()

    # 可视化网格图
    plt.imshow(feature_map_grid_np)
    plt.title('Feature Map Grid')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    #---------------------------
    #如果特征图太小，可双线性插值放大，scale为放大倍数
    scale = 1
    BI = BilinearInterpolation(scale, scale)
    # feature_map = BI.transform(feature_map)


    # 读取保存的张量文件
    tensor1 = torch.load('tensor1.pt')
    tensor2 = torch.load('tensor2.pt')
    print(tensor1.size())
    print(tensor2.size())

    # visualize_channels_separately(tensor1, 1)

    visualize_channels_as_grid(tensor1, 1)
    # visualize_channels_separately(tensor2, 1)


