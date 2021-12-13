"""
这个脚本将其他格式的图片数据转换为 tiff 格式的
"""
import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
from tools.image_3D_io import load_image_3d, save_image_3d

image_suffix = ['.jpg', '.png', '.bmp', '.jpeg']

def to_tiff(image_root_source, image_root_saved):
    file_name_list = os.listdir(image_root_source)
    if not os.path.isdir(image_root_saved):
        os.makedirs(image_root_saved)
    number_image_processed = 0
    for file_name in file_name_list:
        name, suffix = os.path.splitext(file_name)
        if suffix not in image_suffix:
            continue
        file_full_name = os.path.join(image_root_source, file_name)
        image = cv2.imread(file_full_name,0)
        cv2.imwrite(os.path.join(image_root_saved, name) + '.tiff', image)
        number_image_processed += 1
        print('have processed {} image'.format(number_image_processed))

def projection(image, dim, flag = True):
    """
    将图像矩阵 image 在第 dim 维度上进行投影，返回比 image 低一维的图像矩阵
    :param image: 原图像矩阵，np.multiarray
    :param dim: int
    :param flag: bool, 若flag 为真，则消除矩阵的维度值等于1的那些维度
    :return:
    """
    assert dim < len(image.shape)
    image_target = np.sum(image, dim)
    if flag:
        shape_ = image_target.shape
        shape_new = tuple([a for a in shape_ if a != 1])
        image_target = image_target.reshape(shape_new)
    return image_target

def locate_maximum_point(line):
    """
    line 是一个散点列表，本程序将其中的异常极大值点确定出来
    :param line: list, 散点列表
    :return:
    """
    if not isinstance(line, np.ndarray):
        line = np.array(line)
    shape_ = line.shape
    assert line.size == np.max(shape_)
    mean_line = np.mean(line)
    line = np.abs(line - mean_line)
    line = (line > (2.5*np.mean(line))) + 0
    return line

def filtering(image_3d, k, stage = 0):
    """
    这个函数对三维图像矩阵 image_3d 进行滤波
    :param image_3d: 三维图像矩阵，np.multiarray
    注：image_3d 是一个神经元图像，由于采样过程中仪器引入的噪声，使得image_3d图像数据中有明显的人工痕迹
    :param k: int, 邻域大小
    :return:
    """
    assert len(image_3d.shape) == 3
    image_2d = projection(image_3d, 0)
    image_1d = projection(image_2d, 0)
    line = locate_maximum_point(image_1d)
    print([index for index in range(len(line)) if line[index] != 0])
    depth, height, width = image_3d.shape
    assert len(line) == width
    image_3d_ = image_3d
    total = sum(line) * depth * height
    number_now = 0
    for col in range(width):       # the 3th index, x-axis
        if line[col] == 0:
            continue
        for sli in range(depth):
            for row in range(height):
                mean_coor, mean_coor_0, mean_coor_1 = neighbourhood_mean(image_3d, (sli, row, col), k)
                if image_3d[sli, row, col] > mean_coor:
                    if stage == 0:
                        image_3d_[sli, row, col] = mean_coor_0
                    else:
                        image_3d_[sli, row, col] = mean_coor_1
                number_now += 1
                if number_now % 10000 == 0:
                    print('have processed {} persents'.format(100 * number_now / total))
    return image_3d_

def neighbourhood_mean(image_3d, point_coordinate, k=3):
    """
    返回图像 image_3d 在以点 point_coordinate 为中心的 k 邻域范围内的均值
    :param image_3d: np.multiarray
    :param point_coordinate: 中心点，tuple, len(.) = 3
    :param k: int，邻域范围
    :return:
    """
    assert len(image_3d.shape) == 3
    assert len(point_coordinate) == 3
    assert k % 2 == 1
    r = int((k - 1) / 2)
    depth, height, width = image_3d.shape
    s = 0
    s_0 = 0
    s_1 = 0
    for sli in range(point_coordinate[0]-r, point_coordinate[0]+r+1):
        for row in range(point_coordinate[1]-r, point_coordinate[1]+r+1):
            for col in range(point_coordinate[2]-r, point_coordinate[2]+r+1):
                sli_ = sli
                row_ = row
                col_ = col
                if sli < 0: sli_ = -sli
                if row < 0: row_ = -row
                if col < 0: col_ = -col
                if sli >= depth: sli_ = depth - sli - 2
                if row >= height: row_ = height - row - 2
                if col >= width: col_ = width - col - 2
                s += image_3d[sli_, row_, col_]
                if sli == point_coordinate[0]:
                    s_0 += image_3d[sli_, row_, col_]
                if row == point_coordinate[1]:
                    s_1 += image_3d[sli_, row_, col_]
    s_0 = s_0 - image_3d[point_coordinate]
    return s / (k**3), s_0 / (k**2 - 1), s_1 / (k**2 - 1)

def enhance_image_3d(image_3d, ratio = 20, E = 10):
    """
    这个程序使用对数拉伸函数将图像矩阵 image_3d 进行增强
    :param image_3d: np.multiarray
    :param E: int, 对数拉伸函数的指数
    :return:
    """
    mean_ = np.mean(image_3d)
    mean_3d = 100 if mean_ * ratio > 100 else mean_ * ratio
    print(mean_, mean_3d)
    print(image_3d.shape)
    for index, image in enumerate(image_3d):
        shape_ = image.shape
        ve = image.reshape(1,-1)
        ve = [1 + ((mean_3d / (x+0.001)) ** E) for x in ve]
        ve = [(255 / x) for x in ve]
        #ve = (x for x in ve)
        ve = np.array(list(ve), dtype=np.uint8)
        image = ve.reshape(shape_)
        image_3d[index] = image
        if index % 10 == 0:
            print('(enhance_image_3d) -- have processed {} 2D-images'.format(index))
    return image_3d

def threshold_hard(image_3d, threshold = None):
    """
    对三维图像 image_3d 进行硬阈值滤波
    :param image_3d: np.array
    :param threshold: int
    :return:
    """
    threshold = np.mean(image_3d) if threshold == None else threshold
    assert threshold > 0 and threshold <= 255
    image_3d[image_3d < threshold] = 0
    return image_3d

def fft_image_2d(image):
    """
    对二维图像矩阵 image 进行傅里叶变换
    :param image: np.array
    :return:
    """
    image_fft = np.fft.fft2(image)
    image_fft_shift = np.fft.fftshift(image_fft)
    #cv2.imshow('image', image)
    #cv2.imshow('image_fft', np.log(np.abs(image_fft)))
    #cv2.imshow('image_fft_shift', np.log(np.abs(image_fft_shift)))
    #cv2.waitKey(0)
    plt.subplot(121), plt.imshow(image, 'gray'), plt.title('original')
    #plt.subplot(132), plt.imshow(np.log(np.abs(image_fft)), 'gray'), plt.title('fft')
    plt.subplot(122), plt.imshow(np.log(np.abs(image_fft_shift)), 'gray'), plt.title('fft shift')
    plt.show()

def close_image_3d(image_3d, dim = 0, size = 3):
    """
    对三维图像矩阵分片进行闭操作
    :param image_3d: np.array
    :param dim: int
    :param size: 闭操作的核大小
    :return:
    """
    if dim == 0:
        pass
    else:
        shape_ = [image_3d.shape[i] for i in range(len(image_3d.shape)) if i != dim]
        shape_.insert(0, image_3d.shape[dim])
        image_3d = image_3d.reshape(shape_)
    kernel = np.ones((size, size), dtype=np.uint8)
    for index, image in enumerate(image_3d[:]):
        image_3d[index] = cv2.erode(cv2.dilate(image, kernel), kernel)
    return image_3d

def process(image_root, image_root_save):
    """
    这个程序将路径 image_root 中保存的图像经过处理后，保存到 image_root_save 路径下
    处理过程为 滤波 -> 增强 -> 滤波
    :param image_root:
    :param image_root_save:
    :return:
    """
    image_3d = load_image_3d(image_root)
    shape_ = image_3d.shape
    shape_new = tuple([i for i in shape_ if i != 1])
    image_3d.resize(shape_new)
    #image_3d = filtering(image_3d, k=3)
    image_3d = enhance_image_3d(image_3d, ratio=1.3, E = 20)
    #image_3d = filtering(image_3d, k=3)
    #image_3d = threshold_hard(image_3d, threshold = 100)
    #image_3d = close_image_3d(image_3d, dim = 0, size = 3)
    save_image_3d(image_3d, image_root_save, dim=0, suffix='.tiff')

def process_1(image_root, image_root_save):
    """
    这个程序将路径 image_root 中保存的图像经过处理后，保存到 image_root_save 路径下
    处理过程为 滤波 -> 增强 -> 滤波
    :param image_root:
    :param image_root_save:
    :return:
    """
    image_3d = load_image_3d(image_root)
    shape_ = image_3d.shape
    shape_new = tuple([i for i in shape_ if i != 1])
    image_3d.resize(shape_new)
    image_3d = enhance_image_3d(image_3d, ratio=1.3, E=20)
    #image_3d = threshold_hard(image_3d, threshold=100)
    save_image_3d(image_3d, image_root_save, dim=0, suffix='.tiff')

if __name__ == '__main__':
    image_root_source = '/home/li-qiufu/Desktop/AD0603-Ment-L-8751-fiber04/new'
    image_root_saved = '/home/li-qiufu/Desktop/AD0603-Ment-L-8751-fiber04/new_tiff'
    #image_root_saved_1 = '/home/li-qiufu/Downloads/AD0601-Ment-L-6951-fiber01_tiff_enhance'
    to_tiff(image_root_source, image_root_saved)
    #process(image_root = image_root_saved, image_root_save = image_root_saved_1)
