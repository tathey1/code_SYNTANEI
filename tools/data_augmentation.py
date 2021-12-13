"""
    这个脚本实现神经元数据的增强：翻转、加噪；不进行随机裁剪
"""

import os
import random
import numpy as np
from tools.image_3D_io import load_image_3d, save_image_3d

class Neuron_Data_Image():
    """
    这个类型描述神经元图像数据及其标签矩阵，它们都是三维矩阵，并以二维图像序列保存；
    而非之前的重构数据
    """
    def __init__(self, data_path):
        """

        :param data_path: 一个神经元图像数据的路径，该路径下有两个子路径 image、label；
                          image中保存图像数据，label中保存标签矩阵
        """
        self.data_path = data_path
        self.image_data_path = os.path.join(self.data_path, 'image')
        self.label_data_path = os.path.join(self.data_path, 'label')
        self._check()
        self.image_data = load_image_3d(self.image_data_path)
        self.label_data = load_image_3d(self.label_data_path)

    def _check(self):
        """
        检查参数是否符合要求
        :return:
        """
        if not os.path.isdir(self.data_path):
            raise FileExistsError('神经元数据的保存路径不存在')
        if not os.path.isdir(self.image_data_path):
            raise FileExistsError('神经元图像数据的保存路径不存在')
        if not os.path.isdir(self.label_data_path):
            raise FileExistsError('神经元标签数据的保存路径不存在')

    def flip(self, axis = 'x'):
        """
        翻转，只能绕x轴或y轴翻转180°，不绕z轴翻转，也不进行其他角度或方向的旋转；
        显然进行翻转，要同时对图像和标签进行处理
        :param axis: 翻转轴，这里axis取值范围是 x 或 y，绕这个轴翻转180°
        :return:
        """
        if axis.lower() == 'y':
            self.image_data = self.image_data[:,:,::-1]
            self.image_data = self.image_data[::-1,:,:]
            self.label_data = self.label_data[:,:,::-1]
            self.label_data = self.label_data[::-1,:,:]
        elif axis.lower() == 'x':
            self.image_data = self.image_data[:,::-1,:]
            self.image_data = self.image_data[::-1,:,:]
            self.label_data = self.label_data[:,::-1,:]
            self.label_data = self.label_data[::-1,:,:]
        else:
            raise ValueError('axis 参数只能是 x 或 y')

    def add_noise(self, mean = 0, noise_level = 1):
        """
        加噪，对每个图像添加噪声，显然只需对图像数据添加噪声，不需要处理标签
        :param mean = 0: 零均值，不建议更改
        :param noise_level: 噪声水平，图像均值的权重因子，相乘后作为高斯噪声方差
        :return:
        """
        mean_image = np.mean(self.image_data)
        noise_image = np.random.normal(loc = mean, scale = noise_level * mean_image, size = self.image_data.shape)
        self.image_data = self.image_data + noise_image
        self.image_data[self.image_data < 0] = 0
        self.image_data[self.image_data > 255] = 255

    def save_data(self, saved_path):
        """
        保存处理后的数据，包括图像和标签
        :param saved_path:
        :return:
        """
        self.image_saved_path = os.path.join(saved_path, 'image')
        self.label_saved_path = os.path.join(saved_path, 'label')
        save_image_3d(self.image_data, self.image_saved_path)
        save_image_3d(self.label_data, self.label_saved_path)


def main():
    root = '/home/li-qiufu/PycharmProjects/MyDataBase/1_HUST_DATA/HUST_DATA_noise'
    #data_bases = ['DataBase_16', 'DataBase_17', 'DataBase_18', 'DataBase_19', 'DataBase_20', 'DataBase_21']
    angles = ['angle_0', 'angle_120', 'angle_180', 'angle_240', 'angle_300', 'angle_60']

    root_saved = '/home/li-qiufu/PycharmProjects/MyDataBase/1_HUST_DATA/HUST_DATA_noise_x'

    for angle in angles:
        subsubpath = os.path.join(root, angle)
        file_list = os.listdir(subsubpath)
        file_list.sort()
        for file_name in file_list:
            data_path = os.path.join(subsubpath, file_name)
            print('processing {}'.format(data_path))
            NDI = Neuron_Data_Image(data_path = data_path)
            saved_path = os.path.join(root_saved, angle, file_name)
            #NDI.add_noise(mean = 10, noise_level = 10)
            NDI.flip(axis = 'x')
            NDI.save_data(saved_path = saved_path)


if __name__ == '__main__':
    main()