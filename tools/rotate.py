"""
这个脚本将 DataBase_4 中的神经元图像进行相应的旋转，使得其图像尽可能符合 64*256*256(depth*height*width，之后使用时候会缩放至这个大小)
"""

from tools.image_fusion_in_spatial_domain import NeuronImageWithLabel
from tools.image_fusion_in_spatial_domain import NeuronNodeList_SWC, NeuronNodeList
from tools.image_fusion_in_spatial_domain import Image3D_PATH
import os
import math
import numpy as np
import shutil

def cut_all(root_source = '/home/li-qiufu/PycharmProjects/MyDataBase/DataBase_1'):
    object_list = os.listdir(root_source)
    object_list.sort()
    for object_ in object_list:
        print('processing {} ...'.format(object_))
        path = os.path.join(root_source, object_)
        if not os.path.isdir(path):
            continue
        image_path = os.path.join(path, 'image_tiff')
        file_swc = os.path.join(path, object_ + '_0.swc')
        file_swc_save = os.path.join(path, object_ + '_0_cut_all.swc')
        image_with_label = NeuronImageWithLabel(image_path = image_path, file_swc = file_swc)
        image_with_label.cut_whole().save(saved_file_name = file_swc_save)

def copy_from_to(root_source = '/home/li-qiufu/PycharmProjects/MyDataBase/DataBase_1',
                 root_target = '/home/li-qiufu/PycharmProjects/MyDataBase/DataBase_4'):
    object_list = os.listdir(root_target)
    object_list.sort()
    for object_ in object_list:
        print('copying {}'.format(object_))
        path_target = os.path.join(root_target, object_)
        if not os.path.isdir(path_target):
            continue
        path_source_file = os.path.join(root_source, object_, object_ + '_0_cut_all.swc')
        path_target_file = os.path.join(root_target, object_, object_ + '_0.swc')
        shutil.copy(path_source_file, path_target_file)


def rotate_0(root_source, root_target):
    """
    这个程序保证每个神经元的 X 轴方向尺寸最大
    :return:
    """
    object_list = os.listdir(root_source)
    object_list.sort()
    for object_ in object_list:
        print('processing {}'.format(object_))
        if object_ == 'swc':
            continue
        if not os.path.isdir(os.path.join(root_source, object_)):
            continue
        path_source = os.path.join(root_source, object_)
        path_target = os.path.join(root_target, object_)
        image_path = os.path.join(path_source, 'image')
        file_swc = os.path.join(path_source, object_ + '_0.swc')
        image_path_save = os.path.join(path_target, 'image')
        file_swc_save = os.path.join(path_target, object_ + '_0.swc')

        image_with_label = NeuronImageWithLabel(image_path = image_path, file_swc = file_swc)
        depth, height, width = image_with_label.shape()
        angle_z = 0
        angle_y = 0
        angle_x = 0
        if width == max(depth, height, width):
            pass
        elif height == max(depth, height, width):
            angle_z = math.pi / 2
            angle_y = 0
            angle_x = 0
        elif depth == max(depth, height, width):
            angle_z = 0
            angle_y = math.pi / 2
            angle_x = 0

        image_with_label.neuronnodelist.rotate(angle_Z = angle_z,
                                               angle_Y = angle_y,
                                               angle_X = angle_x).save(saved_file_name = file_swc_save)
        swc_noise_list = os.listdir(path_source)
        swc_noise_list.sort()
        for swc_noise in swc_noise_list:
            if 'noise' not in swc_noise:
                continue
            file_swc_noise = os.path.join(path_source, swc_noise)
            file_swc_noise_save = os.path.join(path_target, swc_noise)
            neuron_node_list = NeuronNodeList_SWC(file_swc = file_swc_noise,
                                                  depth = depth,
                                                  height = height,
                                                  width = width)
            neuron_node_list.rotate(angle_Z = angle_z,
                                    angle_Y = angle_y,
                                    angle_X = angle_x).save(saved_file_name = file_swc_noise_save)


def rotate_1(root_source, root_target):
    """
    这个程序保证每个神经元的 Z 轴方向尺寸最小
    :return:
    """
    object_list = os.listdir(root_source)
    object_list.sort()
    for object_ in object_list:
        print('processing {}'.format(object_))
        if object_ == 'swc':
            continue
        if not os.path.isdir(os.path.join(root_source, object_)):
            continue
        path_source = os.path.join(root_source, object_)
        path_target = os.path.join(root_target, object_)
        if not os.path.isdir(path_target):
            os.mkdir(path_target)
        image_path = os.path.join(path_source, 'image')
        file_swc = os.path.join(path_source, object_ + '_0.swc')
        image_path_save = os.path.join(path_target, 'image')
        file_swc_save = os.path.join(path_target, object_ + '_0.swc')

        image_with_label = NeuronImageWithLabel(image_path = image_path, file_swc = file_swc)
        depth, height, width = image_with_label.shape()
        angle_z = 0
        angle_y = 0
        angle_x = 0
        if depth == min(depth, height, width):
            pass
        elif height == min(depth, height, width):
            angle_z = 0
            angle_y = 0
            angle_x = math.pi / 2
        elif width == min(depth, height, width):
            angle_z = 0
            angle_y = math.pi / 2
            angle_x = 0

        image_with_label.neuronnodelist.rotate(angle_Z = angle_z,
                                               angle_Y = angle_y,
                                               angle_X = angle_x).save(saved_file_name = file_swc_save)
        swc_noise_list = os.listdir(path_source)
        swc_noise_list.sort()
        for swc_noise in swc_noise_list:
            if 'noise' not in swc_noise:
                continue
            file_swc_noise = os.path.join(path_source, swc_noise)
            file_swc_noise_save = os.path.join(path_target, swc_noise)
            neuron_node_list = NeuronNodeList_SWC(file_swc = file_swc_noise,
                                                  depth = depth,
                                                  height = height,
                                                  width = width)
            neuron_node_list.rotate(angle_Z = angle_z,
                                    angle_Y = angle_y,
                                    angle_X = angle_x).save(saved_file_name = file_swc_noise_save)


def rotate_2(root_source, root_target, according_info):
    """
    这个程序保证神经元的 Z 轴的尺寸尽量是 X 轴尺寸的 1/4
    :return:
    """
    if not os.path.isdir(root_target):
        os.mkdir(root_target)
    info = open(os.path.join(root_target, 'info'), 'w')
    info.write('from: \n' + root_source + '\nto\n' + root_target + '\n\n\n')

    info_lines = open(according_info).readlines()
    for info_line in info_lines:
        ele = info_line.split(' ')
        object_ = ele[0]
        add_noise = 0
        if float(ele[1]) > 0.6:
            filler = 5 * float(ele[2])
            add_noise = filler
        elif float(ele[2]) < 6:
            filler = 2 * float(ele[2])
        elif float(ele[2]) < 10:
            filler = 1.5 * float(ele[2])
        else:
            filler = 1.05 * float(ele[2])
        path_source = os.path.join(root_source, object_)
        path_target = os.path.join(root_target, object_)
        if not os.path.isdir(path_target):
            os.mkdir(path_target)
        image_path = os.path.join(path_source, 'image')
        file_swc = os.path.join(path_source, object_ + '_0.swc')
        image_path_save = os.path.join(path_target, 'image')
        file_swc_save = os.path.join(path_target, object_ + '_0.swc')

        image_with_label = NeuronImageWithLabel(image_path = image_path, file_swc = file_swc)
        depth, height, width = image_with_label.shape()
        image_add_noise = np.random.random(size = (depth, height, width)) * add_noise
        image_with_label.image3d.image_3d = image_with_label.image3d.image_3d + image_add_noise

        angle_z = 0
        angle_y = 0
        angle_x = 0
        if depth >= 1. * width / 3:
            text = 'processing {}'.format(object_) + ' ---- cp -r ' + path_source + '/* ' + path_target
            print(text)
            info.write(text + '\n')
            image_with_label.neuronnodelist.save(saved_file_name = file_swc_save)
        else:
            angle_z = 0
            angle_y = math.atan(1. / 6)
            angle_x = math.atan(1. / 4)
            text = 'processing {} ---- rotate ({},{}), filler = {}'.\
                format(object_, angle_x * 180 / math.pi, angle_y * 180 / math.pi, filler)
            print(text)
            info.write(text + '\n')
            image_with_label.neuronnodelist.rotate(angle_Z = angle_z,
                                                   angle_Y = angle_y,
                                                   angle_X = angle_x).save(saved_file_name = file_swc_save)
        swc_noise_list = os.listdir(path_source)
        swc_noise_list.sort()
        for swc_noise in swc_noise_list:
            if 'noise' not in swc_noise:
                continue
            file_swc_noise = os.path.join(path_source, swc_noise)
            file_swc_noise_save = os.path.join(path_target, swc_noise)
            neuron_node_list = NeuronNodeList_SWC(file_swc = file_swc_noise,
                                                  depth = depth,
                                                  height = height,
                                                  width = width)
            neuron_node_list.rotate(angle_Z = angle_z,
                                    angle_Y = angle_y,
                                    angle_X = angle_x).save(saved_file_name = file_swc_noise_save)
    info.close()


def resize_0(root_source, root_target):
    """
    这个程序保证神经元的 X 轴尺寸不超过500
    :return:
    """
    object_list = os.listdir(root_source)
    object_list.sort()
    for object_ in object_list:
        if object_ == 'swc':
            continue
        if not os.path.isdir(os.path.join(root_source, object_)):
            continue
        path_source = os.path.join(root_source, object_)
        path_target = os.path.join(root_target, object_)
        if not os.path.isdir(path_target):
            os.mkdir(path_target)
        image_path = os.path.join(path_source, 'image')
        file_swc = os.path.join(path_source, object_ + '_0.swc')
        image_path_save = os.path.join(path_target, 'image')
        file_swc_save = os.path.join(path_target, object_ + '_0.swc')

        image_with_label = NeuronImageWithLabel(image_path = image_path, file_swc = file_swc)
        depth, height, width = image_with_label.shape()

        if width <= 500:
            print('processing {}'.format(object_) + ' ---- cp -r ' + path_source + '/* ' + path_target)
            os.system('cp -r ' + path_source + '/* ' + path_target)
        else:
            width_new = 500
            height_new = round(height * 500 / width)
            depth_new = depth if depth < width_new else round(depth * 500 / width)
            print('processing {} ---- resize ({},{},{})'.format(object_, depth_new, height_new, width_new))
            image_with_label.neuronnodelist.resize(shape_new = (depth_new, height_new, width_new))\
                .save(saved_file_name = file_swc_save)
            swc_noise_list = os.listdir(path_source)
            swc_noise_list.sort()
            for swc_noise in swc_noise_list:
                if 'noise' not in swc_noise:
                    continue
                file_swc_noise = os.path.join(path_source, swc_noise)
                file_swc_noise_save = os.path.join(path_target, swc_noise)
                neuron_node_list = NeuronNodeList_SWC(file_swc = file_swc_noise,
                                                      depth = depth,
                                                      height = height,
                                                      width = width)
                neuron_node_list.resize(shape_new = (depth_new, height_new, width_new))\
                    .save(saved_file_name = file_swc_noise_save)


def cut_whole_0(root_source, root_target):
    """
    这个程序将旋转后的神经元进行裁剪
    :param root_source:
    :param root_target:
    :return:
    """
    object_list = os.listdir(root_source)
    object_list.sort()
    if not os.path.isdir(root_target):
        os.mkdir(root_target)
    info = open(os.path.join(root_target, 'info'), 'w')
    info.write('cut_whole_0\n' + 'from\n' + ' '*4  + root_source + '\n' + 'to\n' + ' '*4 + root_target + '\n\n\n')
    for object_ in object_list:
        path_source = os.path.join(root_source, object_)
        if not os.path.isdir(path_source):
            continue
        path_target = os.path.join(root_target, object_)
        if not os.path.isdir(path_target):
            os.mkdir(path_target)
        image_path = os.path.join(path_source, 'image')
        file_swc = os.path.join(path_source, object_ + '_0.swc')
        image_path_save = os.path.join(path_target, 'image')
        file_swc_save = os.path.join(path_target, object_ + '_0.swc')

        image_with_label = NeuronImageWithLabel(image_path = image_path, file_swc = file_swc)
        depth, height, width = image_with_label.shape()
        info.write('processing {}\n'.format(object_))
        print('processing {}'.format(object_))
        image_with_label.neuronnodelist, coor = image_with_label.neuronnodelist.cut_whole()
        image_with_label.image3d = image_with_label.image3d.cut(shape = image_with_label.neuronnodelist.shape(),
                                                                coor = coor)
        image_with_label.refresh_shape()
        image_with_label.save(image_save_root = image_path_save, saved_file_name = file_swc_save)
        swc_noise_list = os.listdir(path_source)
        swc_noise_list.sort()
        for swc_noise in swc_noise_list:
            if 'noise' not in swc_noise:
                continue
            file_swc_noise = os.path.join(path_source, swc_noise)
            file_swc_noise_save = os.path.join(path_target, swc_noise)
            neuron_node_list = NeuronNodeList_SWC(file_swc = file_swc_noise,
                                                  depth = depth,
                                                  height = height,
                                                  width = width)
            neuron_node_list.cut_as_shape(shape = image_with_label.shape(),
                                          coor = coor)[0].save(saved_file_name = file_swc_noise_save)


def rotate_3(root_source, root_target, according_info, angles = None):
    """
    这个程序保证神经元的 Z 轴的尺寸尽量是 X 轴尺寸的 1/4
    :return:
    """
    angles = [60, 120, 180, 240, 300] if angles == None else angles
    if not os.path.isdir(root_target):
        os.mkdir(root_target)
    info = open(os.path.join(root_target, 'info'), 'w')
    info.write('rotate_3\n')
    info.write('from\n' + ' '*4 + root_source + '\n' + 'to\n' + ' ' * 4 + root_target + '\n\n\n')
    info_lines = open(according_info).readlines()
    for angle in angles:
        if not os.path.isdir(os.path.join(root_target, 'angle_' + str(angle))):
            os.mkdir(os.path.join(root_target, 'angle_' + str(angle)))
        for info_line in info_lines:
            ele = info_line.split(' ')
            object_ = ele[0]
            if float(ele[1]) > 0.6:
                filler = 5 * float(ele[2])
            elif float(ele[2]) < 6:
                filler = 2 * float(ele[2])
            elif float(ele[2]) < 10:
                filler = 1.5 * float(ele[2])
            else:
                filler = 1.05 * float(ele[2])
            path_source = os.path.join(root_source, object_)
            path_target = os.path.join(root_target, 'angle_' + str(angle), object_)
            if not os.path.isdir(path_target):
                os.mkdir(path_target)
            image_path_source = os.path.join(path_source, 'image')
            file_swc_source = os.path.join(path_source, object_ + '_0.swc')
            image_path_target = os.path.join(path_target, 'image')
            file_swc_target = os.path.join(path_target, object_ + '_0.swc')

            image_with_label_source = NeuronImageWithLabel(image_path = image_path_source,
                                                           file_swc = file_swc_source)

            depth, height, width = image_with_label_source.shape()
            angle_z = math.pi * angle / 180
            angle_y = 0
            angle_x = 0
            print('angle = {}, processing {} ---- rotate ({})'.format(angle, object_, angle_z * 180 / math.pi))
            info.write('angle = {}, processing {} ---- rotate ({})\n'.format(angle, object_, angle_z * 180 / math.pi))
            image_with_label_source.neuronnodelist.rotate(angle_Z = angle_z,
                                                          angle_Y = angle_y,
                                                          angle_X = angle_x,).save(saved_file_name = file_swc_target)
            swc_noise_list = os.listdir(path_source)
            swc_noise_list.sort()
            for swc_noise in swc_noise_list:
                if 'noise' not in swc_noise:
                    continue
                file_swc_noise = os.path.join(path_source, swc_noise)
                file_swc_noise_save = os.path.join(path_target, swc_noise)
                neuron_node_list = NeuronNodeList_SWC(file_swc = file_swc_noise,
                                                      depth = depth,
                                                      height = height,
                                                      width = width)
                neuron_node_list.rotate(angle_Z = angle_z,
                                        angle_Y = angle_y,
                                        angle_X = angle_x).save(saved_file_name = file_swc_noise_save)


def compute_zero_ratio(root_source):
    """
    计算神经元图像中 0 元素所占的比例
    :param root_source:
    :return:
    """
    info = open(os.path.join(root_source, 'zero_ratio.info'), 'w')
    object_list = os.listdir(root_source)
    object_list.sort()
    for object_ in object_list:
        if not os.path.isdir(os.path.join(root_source, object_)):
            continue
        image_path = os.path.join(root_source, object_, 'image')
        image_neuron = Image3D_PATH(image_path = image_path)
        mean_value = np.mean(image_neuron.image_3d)
        zero_m = (image_neuron.image_3d == 0) + 0
        ratio = np.sum(zero_m) / zero_m.size
        text = object_ + ' ' + str(ratio) + ' ' + str(mean_value)
        info.write(text + '\n')
        print(text)
    info.close()


def change_type(file_swc = '/home/li-qiufu/PycharmProjects/MyDataBase/DataBase_4/000060/000060_0.swc'):
    neuron_node_list = NeuronNodeList_SWC(file_swc = file_swc, depth = 76, height = 508, width = 308)
    for key in neuron_node_list._neuron_node_list.keys():
        node = neuron_node_list._neuron_node_list[key]
        if node.type == '1':
            node.type = '3'
    neuron_node_list.save(saved_file_name = file_swc)

def change_snake_data(root_source):
    """
    snake 追踪方法得到的数据的节点半径全是 2，过大，这个程序将他们都改成 1
    :param root_source:
    :return:
    """
    info = open(os.path.join(root_source, 'info_snake'), 'w')
    info.write('change_snake_data\n')
    info.write('root_source: {}\n\n\n'.format(root_source))
    object_list = os.listdir(root_source)
    object_list.sort()
    for object_ in object_list:
        path_source = os.path.join(root_source, object_)
        if not os.path.isdir(path_source):
            continue
        print('processing {}'.format(object_))
        info.write('processing {}\n'.format(object_))
        image_path = os.path.join(path_source, 'image')
        image_3d = Image3D_PATH(image_path = image_path)
        noise_swc_list = os.listdir(path_source)
        noise_swc_list.sort()
        for noise_swc in noise_swc_list:
            if 'noise' not in noise_swc:
                continue
            file_swc = os.path.join(path_source, noise_swc)
            neuron_node_list = NeuronNodeList_SWC(file_swc = file_swc,
                                                  depth = image_3d.depth,
                                                  width = image_3d.width,
                                                  height = image_3d.height)
            for key in neuron_node_list.keys():
                node = neuron_node_list._neuron_node_list[key]
                if node.type == '2' and node.radius == 2:
                    node.radius = 1
            neuron_node_list.save(saved_file_name = file_swc)


if __name__ == '__main__':
    pass
