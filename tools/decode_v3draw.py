"""
这个脚本将 vaa3d 能够识别的 v3draw 格式的图像数据转换成一张张图片序列
"""

import struct
import os
import numpy as np
import cv2

def decode_image_v3draw(file_name, image_root_save = None, byteorder='little', height = None, width = None, offset = 27):
    """
    从名为 file_name 的 v3draw 文件中解码数据，分片读取，每片数据的高宽为 height、width，并将解码得到的图像数据保存在 image_root_save 中
    :param file_name: 二进制文件名
    :param image_root_save: 图像保存路径
    :param height: 高
    :param width: 宽
    # :param nbytes: 每个数据的字节长度
    :param byteorder: 'big' or 'little'
    :return:
    """
    assert os.path.isfile(file_name) and file_name.endswith('v3draw')
    if image_root_save == None:
        raise ValueError('没有制定解码后图片保存路径')
    if not os.path.isdir(image_root_save):
        os.makedirs(image_root_save)
    f = open(os.path.join(file_name), 'rb')
    records = f.read()
    f.close()

    if byteorder == 'big' or byteorder == '>' or byteorder == '!':
        byteorder = '>'
    elif byteorder == 'little' or byteorder == '<':
        byteorder = '<'

    width_, height_, depth, channel = struct.unpack_from(byteorder + 'IIII', records, offset)
    height = height_ if height == None else height
    print(width_, height_, depth, channel)
    width = width_ if width == None else width
    offset += 16

    current = 0
    nslice = 0
    row = 0
    col = 0
    slice = np.zeros((height, width))
    pixel = struct.unpack_from(byteorder + 'B', records, offset)[0]

    while True:
        offset += 1
        current += 1
        slice[height - row - 1,col] = pixel
        if col == (width - 1) and row == (height - 1):
            #print(slice)
            #cv2.imshow('neuron_image', slice)
            #cv2.waitKey(0)
            col = 0
            row = 0
            current = 0
            cv2.imwrite(os.path.join(image_root_save, str(nslice).zfill(6)) + '.tiff', slice)
            nslice += 1
            print('have got {} images, current = {}'.format(nslice, current))
            slice = np.zeros((height, width))
        elif col == (width - 1):
            col = 0
            row += 1
        else:
            col += 1
        try:
            pixel = struct.unpack_from(byteorder + 'B', records, offset)[0]
        except:
            print('have got {} images, current = {}'.format(nslice, current))
            break

def main():
    image_root = '/home/li-qiufu/PycharmProjects/MyDataBase/DataBase_1'
    for i in range(96):
        sub_path = os.path.join(image_root, str(i).zfill(6))
        file_name = os.path.join(sub_path, str(i).zfill(6) + '.v3dpbd.v3draw')
        image_root_save = os.path.join(sub_path, 'image_tiff')
        if not os.path.isdir(image_root_save):
            os.mkdir(image_root_save)
        print('\n\n    processing {}\n\n'.format(str(i).zfill(6) + '.v3dpbd.v3draw'))
        decode_image_v3draw(file_name=file_name, image_root_save=image_root_save)


if __name__ == '__main__':
    main()