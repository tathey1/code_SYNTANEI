"""
这个脚本根据 DataBase_4 中的文件 id_cut.info 从其中的神经元图像数据中裁剪出相应的神经纤维
"""

from tools.image_fusion_in_spatial_domain import NeuronImageWithLabel, Image3D_PATH, Image3D
import os
import shutil
import numpy as np
from numpy import cov
import matplotlib.pyplot as plt
from copy import deepcopy


def cut_branch(data_source, data_target, cut_info):
    if not os.path.isdir(data_target):
        os.mkdir(data_target)
    info_list = [line.strip() for line in open(cut_info)]
    for line in info_list:
        ele = line.split(' ')
        sub_name = ele[0]
        print('\n\nprocessing {}\n'.format(sub_name))
        if not os.path.isdir(os.path.join(data_target, sub_name)):
            os.mkdir(os.path.join(data_target, sub_name))

        image_path = os.path.join(data_source, sub_name, 'image')
        file_swc = os.path.join(data_source, sub_name, sub_name + '_0.swc')
        image_with_label = NeuronImageWithLabel(image_path=image_path, file_swc=file_swc, label=False)
        for id in ele[1:]:
            print(' ' * 4 + '----' + 'id = {}'.format(id))
            imagelabel_cut = image_with_label.cut_as_swc(id = id)
            if imagelabel_cut.image3d.depth > 2 and imagelabel_cut.width > 10 and imagelabel_cut.height > 10:
                if not os.path.isdir(os.path.join(data_target, sub_name, 'id_'+id)):
                    os.mkdir(os.path.join(data_target, sub_name, 'id_'+id))
                image_save_root = os.path.join(data_target, sub_name, 'id_'+id, 'image')
                file_name = os.path.join(data_target, sub_name, 'id_'+id, 'id_'+id + '.swc')
                imagelabel_cut.save(image_save_root=image_save_root, saved_file_name=file_name)


def sort_branch(root_source, root_target):
    number = 0
    object_list = os.listdir(root_source)
    object_list.sort()
    info = open(os.path.join(root_target, 'info'), 'w')
    info.write('sort_branch\n')
    info.write('from\n' + ' '*4 + root_source + '\n' + 'to\n' + ' '*4 + root_target + '\n\n')
    for object_ in object_list:
        print('\nprocessing {}'.format(object_))
        sub_path = os.path.join(root_source, object_)
        if not os.path.isdir(sub_path):
            continue
        sub_object_list = os.listdir(sub_path)
        for sub_object_ in sub_object_list:
            print(' ' * 4 + '---- ' + sub_object_)
            text = os.path.join(object_, sub_object_) + ' ' + 'branch_' + str(number).zfill(6)
            info.write(text + '\n')
            sub_root_target = os.path.join(root_target, 'branch_' + str(number).zfill(6))
            if not os.path.isdir(sub_root_target):
                os.mkdir(sub_root_target)
            image_data_souce = os.path.join(sub_path, sub_object_, 'image')
            os.system('cp -r ' + image_data_souce + ' ' + sub_root_target)
            swc_data_souce = os.path.join(sub_path, sub_object_, sub_object_ + '.swc')
            swc_data_target = os.path.join(sub_root_target, 'branch_' + str(number).zfill(6) + '.swc')
            shutil.copy(swc_data_souce, swc_data_target)
            number += 1
    info.close()

def stretching_value(root_source, according_info = '/home/li-qiufu/PycharmProjects/MyDataBase/Neuron_Branch/mean_maximum.info'):
    """
    这个程序将图像数据的像素值进行线性拉伸，将其最大值扩大成 254
    :param root_source:
    :return:
    """
    object_list = open(according_info).readlines()
    for line in object_list:
        ele = line.split(' ')
        object_ = ele[0]
        mean_value = float(ele[1])
        maximum_value = float(ele[2])
        if mean_value > 85 and maximum_value > 250:
            print('do nothing for {}'.format(object_))
            continue
        root_path = os.path.join(root_source, object_)
        if not os.path.isdir(root_path):
            continue
        image_path = os.path.join(root_path, 'image')
        image_3D = Image3D_PATH(image_path = image_path)
        ratio_0 = 85 / mean_value
        ratio_1 = 250 / maximum_value
        ratio = max(ratio_0, ratio_1)
        print('processed {}'.format(object_))
        for zz in range(image_3D.depth):
            for yy in range(image_3D.height):
                for xx in range(image_3D.width):
                    value = image_3D.image_3d[zz,yy,xx]
                    image_3D.image_3d[zz,yy,xx] = min(value * ratio, 255)
        image_3D.save(image_save_root = image_path)


def plot_mean(root_source):
    """
    这个程序将图像数据的像素值进行线性拉伸，将其最大值扩大成 254
    :param root_source:
    :return:
    """
    object_list = os.listdir(root_source)
    object_list.sort()
    mean_list = []
    maximum_list = []
    info = open(os.path.join(root_source, 'mean_maximum.info'), 'w')
    for object_ in object_list:
        root_path = os.path.join(root_source, object_)
        if not os.path.isdir(root_path):
            continue
        print('processing {}'.format(object_))
        image_path = os.path.join(root_path, 'image')
        file_swc = os.path.join(root_path, object_ + '.swc')
        image_with_label = NeuronImageWithLabel(image_path = image_path, file_swc = file_swc, label = True)
        sum_ = 0
        number = 0
        maximum = 0
        for zz in range(image_with_label.depth):
            for yy in range(image_with_label.height):
                for xx in range(image_with_label.width):
                    if image_with_label.label_3d.image_3d[zz,yy,xx] != 0:
                        number += 1
                        sum_ += image_with_label.image3d.image_3d[zz,yy,xx]
                        if maximum < image_with_label.image3d.image_3d[zz,yy,xx]:
                            maximum = image_with_label.image3d.image_3d[zz,yy,xx]
        mean_list.append(sum_ / number)
        maximum_list.append(maximum)
        info.write(object_ + ' ' + str(sum_ / number) + ' ' + str(maximum) + '\n')
    #mean_list.sort()
    info.close()
    plt.scatter(range(len(mean_list)), mean_list)
    plt.scatter(range(len(mean_list)), maximum_list)
    plt.show()

def plot_size(root_source):
    object_list = os.listdir(root_source)
    object_list.sort()
    size_list = []
    size_xy_list = []
    x_list = []
    for object_ in object_list:
        root_path = os.path.join(root_source, object_)
        if not os.path.isdir(root_path):
            continue
        x_list.append(int(object_))
        print('processing {}'.format(object_))
        image_path = os.path.join(root_path, 'image')
        file_swc = os.path.join(root_path, object_ + '_0.swc')
        image_with_label = NeuronImageWithLabel(image_path = image_path, file_swc = file_swc, label = False)
        size_list.append(image_with_label.size())
        size_xy_list.append(image_with_label.width * image_with_label.height)
    plt.subplot(211)
    plt.plot(x_list, size_list, c = 'b', marker = '*')
    plt.subplot(212)
    plt.plot(x_list, size_xy_list, c = 'r', marker = 'x')
    plt.show()


def extract_12(label_root):
    """
    将label_3d中的1和2分别提取出来，生成两个label_3d并保存
    :param label_root:
    :return:
    """
    label_root_1 = os.path.join(label_root, '1')
    if not os.path.isdir(label_root_1):
        os.mkdir(label_root_1)
    label_root_2 = os.path.join(label_root, '2')
    if not os.path.isdir(label_root_2):
        os.mkdir(label_root_2)
    image3d = Image3D_PATH(image_path = label_root)
    image3d_1 = Image3D()
    image3d_1.image_3d = deepcopy(image3d.image_3d)
    image3d_1.image_3d[image3d_1.image_3d != 1] = 0
    image3d_1.refresh_shape()
    image3d_2 = Image3D()
    image3d_2.image_3d = deepcopy(image3d.image_3d)
    image3d_2.image_3d[image3d_2.image_3d != 2] = 0
    image3d_2.refresh_shape()
    image3d_1.save(label_root_1)
    image3d_2.save(label_root_2)


def plot_mean_max_swc(root = '/home/li-qiufu/PycharmProjects/MyDataBase/DataBase_15'):
    """
    根据swc文件
    :param root:
    :return:
    """
    r_offset = 2
    info = open(os.path.join(root, 'info'), 'w')
    angles = ['angle_0', 'angle_60', 'angle_120', 'angle_180', 'angle_240', 'angle_300']
    for angle in angles:
        object_root = os.path.join(root, angle)
        if not os.path.isdir(object_root):
            continue
        object_list = os.listdir(object_root)
        object_list.sort()
        for object_ in object_list:
            object_path = os.path.join(object_root, object_)
            if not os.path.isdir(object_path):
                continue
            image_path = os.path.join(object_path, 'image')
            file_swc = os.path.join(object_path, object_ + '_0.swc')
            neuron_image_with_label = NeuronImageWithLabel(image_path = image_path, file_swc = file_swc, label = True)
            sumimum = 0
            maximum = 0
            number = 0
            for zz in range(neuron_image_with_label.depth):
                for yy in range(neuron_image_with_label.height):
                    for xx in range(neuron_image_with_label.width):
                        value = neuron_image_with_label.image3d.image_3d[zz,yy,xx]
                        if neuron_image_with_label.label_3d.image_3d[zz,yy,xx] != 0:
                            number += 1
                            maximum = max(maximum, value)
                            sumimum += value
            print(os.path.join(angle, object_) + ' ' + str(maximum) + ' ' + str(sumimum / number))
            info.write(os.path.join(angle, object_) + ' ' + str(maximum) + ' ' + str(sumimum / number) + '\n')
    info.close()

def stretching_value_new(root, root_save, according_info = '/home/li-qiufu/PycharmProjects/MyDataBase/DataBase_15/info'):
    """
    这个程序将图像数据的像素值进行线性拉伸
    :param root:
    :return:
    """
    angles = ['angle_60', 'angle_120', 'angle_180', 'angle_240', 'angle_300']
    object_list = open(according_info).readlines()
    for angle in angles:
        object_root = os.path.join(root, angle)
        if not os.path.isdir(object_root):
            continue
        object_root_save = os.path.join(root_save, angle)
        if not os.path.isdir(object_root_save):
            os.mkdir(object_root_save)
        for line in object_list:
            ele = line.split(' ')
            _, object_ = os.path.split(ele[0])
            print('processing {}'.format(os.path.join(object_root, object_)))
            maximum_value = float(ele[1])
            mean_value = float(ele[2])
            if mean_value > 170 and maximum_value > 250:
                print('do nothing for {}'.format(object_))
                continue
            root_path = os.path.join(object_root, object_)
            root_path_save = os.path.join(object_root_save, object_)
            if not os.path.isdir(root_path):
                continue
            image_path = os.path.join(root_path, 'image')
            file_swc = os.path.join(root_path, object_ + '_0.swc')
            image_path_save = os.path.join(root_path_save, 'image')
            if not os.path.isdir(image_path_save):
                os.makedirs(image_path_save)
            image_with_label = NeuronImageWithLabel(image_path = image_path, file_swc = file_swc)
            image_according = np.zeros(image_with_label.shape())
            ratio = 170 / mean_value
            r_offset = 2
            keys = list(image_with_label.neuronnodelist._neuron_node_list.keys())
            while keys != []:
                key = keys[0]
                while key:
                    neuron_node = image_with_label.neuronnodelist._neuron_node_list[key]
                    if neuron_node.processed == 1 and neuron_node.child_id == []:
                        keys.remove(key)
                        break
                    elif neuron_node.processed == 1 and neuron_node.child_id != []:
                        key = image_with_label.neuronnodelist._neuron_node_list[key].child_id.pop(0)
                        continue
                    elif neuron_node.processed == 0:
                        r_reset = r_offset + neuron_node.radius
                        points = neuron_node.get_around_points(r_reset = r_reset, shape = image_with_label.shape())
                        for point in points:
                            if image_according[point] == 0:
                                value = image_with_label.image3d.image_3d[point]
                                image_with_label.image3d.image_3d[point] = min(value * ratio, 255)
                                image_according[point] = 1
                        image_with_label.neuronnodelist._neuron_node_list[key].processed = 1
                        if neuron_node.child_id != []:
                            neuron_node_child = image_with_label.neuronnodelist._neuron_node_list[neuron_node.child_id[0]]
                            image_with_label.neuronnodelist._neuron_node_list[key].child_id = neuron_node.child_id
                            points = neuron_node.get_connect_points(neuron_node_child,
                                                                    shape = image_with_label.shape(),
                                                                    r_offset = r_offset)
                            for point in points:
                                if image_according[point] == 0:
                                    value = image_with_label.image3d.image_3d[point]
                                    image_with_label.image3d.image_3d[point] = min(value * ratio, 255)
                                    image_according[point] = 1
                            key = neuron_node.child_id[0]
                            neuron_node.child_id.pop(0)
                            continue
                        else:
                            keys.remove(key)
                            break
            image_with_label.save(image_save_root = image_path_save)

def copy(source_root, target_root):
    angles = ['angle_0', 'angle_60', 'angle_120', 'angle_180', 'angle_240', 'angle_300']
    for angle in angles:
        object_root = os.path.join(source_root, angle)
        object_root_save = os.path.join(target_root, angle)
        object_list = os.listdir(object_root)
        for object_ in object_list:
            root_path = os.path.join(object_root, object_)
            print('processing {} ...'.format(root_path))
            if not os.path.isdir(root_path):
                continue
            root_path_save = os.path.join(object_root_save, object_)
            #label_path = os.path.join(root_path, 'label')
            #os.system('cp -r ' + label_path + ' ' + root_path_save)
            file_list = os.listdir(root_path)
            for file_name in file_list:
                if 'swc' not in file_name:
                    continue
                file_name_full = os.path.join(root_path, file_name)
                os.system('cp ' + file_name_full + ' ' + root_path_save)



if __name__ == '__main__':
    #root_0 = '/home/li-qiufu/PycharmProjects/MyDataBase/neuron_data_0/DataBase_19'
    #root_save_0 = '/home/li-qiufu/PycharmProjects/MyDataBase/neuron_data_1/DataBase_19'
    #copy(root_0, root_save_0)
    #root_1 = '/home/li-qiufu/PycharmProjects/MyDataBase/neuron_data_0/DataBase_20'
    #root_save_1 = '/home/li-qiufu/PycharmProjects/MyDataBase/neuron_data_1/DataBase_20'
    #copy(root_1, root_save_1)
    #root_2 = '/home/li-qiufu/PycharmProjects/MyDataBase/neuron_data_0/DataBase_21'
    #root_save_2 = '/home/li-qiufu/PycharmProjects/MyDataBase/neuron_data_1/DataBase_21'
    #copy(root_2, root_save_2)
    root = '/home/li-qiufu/PycharmProjects/MyDataBase/DataBase_15'
    root_save = '/home/li-qiufu/PycharmProjects/MyDataBase/DataBase_15_enhance'
    stretching_value_new(root, root_save)