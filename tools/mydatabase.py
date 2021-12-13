"""
这个脚本用于处理 ~/MyDataBase 中的神经元图像数据
"""
import os, shutil
from tools.image_fusion_in_spatial_domain import NeuronImageWithLabel
import numpy as np
from tools.decode_v3draw import decode_image_v3draw


image_root = '/home/li-qiufu/PycharmProjects/MyDataBase/DataBase_1'

def main():
    sub_root_list = os.listdir(image_root)
    sub_root_list.sort()
    for sub_root in sub_root_list:
        sub_path = os.path.join(image_root, sub_root)
        if not os.path.isdir(sub_path):
            continue
        image_path = os.path.join(sub_path, 'image_tiff')
        file_swc_name = os.path.join(sub_path, sub_root + '_0.swc')
        image_save_root = os.path.join(sub_path, 'image_cut_whole')
        saved_file_name = os.path.join(sub_path, sub_root + '_0_cut_whole.swc')
        print('\n\n processing \n -------- {} \n     and \n--------{} \n\n'.format(image_path, file_swc_name))
        neuron_image_with_label = NeuronImageWithLabel(image_path = image_path, file_swc = file_swc_name)
        neuron_image_with_label.cut_whole(label=False).save(image_save_root=image_save_root, saved_file_name=saved_file_name)

def main0():
    image_path = '/home/li-qiufu/PycharmProjects/MyDataBase/DataBase_1/000020/image_tiff'
    file_swc_name = '/home/li-qiufu/PycharmProjects/MyDataBase/DataBase_1/000020/000020_0.swc'
    image_save_root = '/home/li-qiufu/PycharmProjects/MyDataBase/DataBase_1/000020/image_cut_whole'
    saved_file_name = '/home/li-qiufu/PycharmProjects/MyDataBase/DataBase_1/000020/000020_0_cut_whole.swc'
    neuron_image_with_label = NeuronImageWithLabel(image_path=image_path, file_swc=file_swc_name)
    neuron_image_with_label.cut_whole().save(image_save_root=image_save_root, saved_file_name=saved_file_name)

def main1():
    image_root_source = '/home/li-qiufu/PycharmProjects/MyDataBase/DataBase_1'
    image_root_target = '/home/li-qiufu/PycharmProjects/MyDataBase/DataBase_2'

    for i in range(96):
        sub_root_source = os.path.join(image_root_source, str(i).zfill(6))
        sub_root_target = os.path.join(image_root_target, str(i).zfill(6))
        if not os.path.isdir(sub_root_source):
            raise ValueError
        if not os.path.isdir(sub_root_target):
            os.mkdir(sub_root_target)
        source_1 = os.path.join(sub_root_source, 'image_cut_whole')
        os.system('cp -r ' + source_1 + ' ' + sub_root_target)
        source_2 = os.path.join(sub_root_source, str(i).zfill(6) + '_0_cut_whole.swc')
        os.system('cp ' + source_2 + ' ' + sub_root_target)

from tools.to_tiff import enhance_image_3d
from image_3D_io import load_image_3d, save_image_3d
def main2():
    image_root_source = '/home/li-qiufu/PycharmProjects/MyDataBase/DataBase_2'
    image_root_target = '/home/li-qiufu/PycharmProjects/MyDataBase/DataBase_3'
    for i in range(96):
        if not os.path.isdir(os.path.join(image_root_target, str(i).zfill(6))):
            os.mkdir(os.path.join(image_root_target, str(i).zfill(6)))
        source_2 = os.path.join(image_root_source, str(i).zfill(6), str(i).zfill(6) + '_0_cut_whole.swc')
        target_2 = os.path.join(image_root_target, str(i).zfill(6), str(i).zfill(6) + '.swc')
        shutil.copy(source_2, target_2)

        print('\n\n---- processing {}\n'.format(str(i).zfill(6)))
        sub_root_source = os.path.join(image_root_source, str(i).zfill(6), 'image_cut_whole')
        print(sub_root_source)
        image_3d = load_image_3d(sub_root_source)
        shape_ = [image_3d.shape[i] for i in range(len(image_3d.shape)) if image_3d.shape[i] != 1]
        image_3d = image_3d.reshape(shape_)
        image_3d = enhance_image_3d(image_3d=image_3d)
        sub_root_target = os.path.join(image_root_target, str(i).zfill(6), 'image')
        save_image_3d(image_3d=image_3d, image_save_root=sub_root_target)

def main3():
    image_root_source = '/home/li-qiufu/PycharmProjects/MyDataBase/DataBase_2'
    image_root_target = '/home/li-qiufu/PycharmProjects/MyDataBase/DataBase_3'
    sub_root_list = os.listdir(image_root_source)
    for sub_root in sub_root_list:
        sub_path_source = os.path.join(image_root_source, sub_root)
        sub_path_target = os.path.join(image_root_target, sub_root)
        if os.path.isdir(sub_path_target):
            continue
        else:
            os.mkdir(sub_path_target)
        source_2 = os.path.join(sub_path_source, sub_root + '_0_cut_whole.swc')
        target_2 = os.path.join(sub_path_target, sub_root + '.swc')
        shutil.copy(source_2, target_2)

        print('\n\n---- processing {}\n'.format(sub_root))
        sub_image_source = os.path.join(sub_path_source, 'image_cut_whole')
        sub_image_target = os.path.join(sub_path_target, 'image')
        if not os.path.isdir(sub_image_target):
            os.mkdir(sub_image_target)
        os.system('cp ' + sub_image_source + '/*' + ' ' + sub_image_target)


def erasing(image_path, file_swc, image_path_save, r_offset = 10):
    """
    将 image_path 和 file_swc 对应的 NeuronImageWithLabel 对象的图像数据中的神经元部分抹除， r_offset 是抹除范围
    :param image_path: NeuronImageWithLabel 的图像数据保存路径
    :param file_swc: NeuronImageWithLabel 的 swc 文件名
    :param image_path_save: 抹除后的图像数据保存路径
    :param r_offset: 抹除范围，对 swc 文件中每个神经元节点的半径进行重置的增量值
    :return:
    """
    image_with_label = NeuronImageWithLabel(image_path=image_path, file_swc=file_swc, label = False)
    #image_with_label.label_3d = image_with_label.neuronnodelist.generate_label_3d(r_offset=r_offset)
    mean_label = np.mean(image_with_label.image3d.image_3d)
    print(mean_label)
    image_3d = image_with_label.image3d.image_3d
    for index, key in enumerate(image_with_label.neuronnodelist.keys()):
        node = image_with_label.neuronnodelist._neuron_node_list[key]
        points = node.get_around_points(r_reset = node.radius + r_offset, shape=image_with_label.shape())
        for point in points:
            image_3d[point] = 0
        print('have process {} node'.format(index))
    """
    for z in range(image_with_label.depth):
        for y in range(image_with_label.height):
            for x in range(image_with_label.width):
                if image_with_label.label_3d.image_3d[z,y,x] == 0:
                    continue
                else:
                    image_3d[z,y,x] = mean_label
        print('have process {} slices'.format(z))
    """
    save_image_3d(image_3d, image_save_root = image_path_save)

def erasing_back(image_path, file_swc, image_path_save, r_offset = 10.):
    """
    将 image_path 和 file_swc 对应的 NeuronImageWithLabel 对象的图像数据中的背景部分抹除， r_offset 是保护范围
    :param image_path: NeuronImageWithLabel 的图像数据保存路径
    :param file_swc: NeuronImageWithLabel 的 swc 文件名
    :param image_path_save: 抹除后的图像数据保存路径
    :param r_offset: 保护范围，对 swc 文件中每个神经元节点的半径进行重置的增量值
    :return:
    """
    image_with_label = NeuronImageWithLabel(image_path=image_path, file_swc=file_swc, label = False)
    image_with_label.label_3d = image_with_label.neuronnodelist.generate_label_3d(r_offset=r_offset)
    mean_label = np.mean(image_with_label.image3d.image_3d)
    print(mean_label)
    image_3d = np.ones(image_with_label.shape()) * mean_label
    for z in range(image_with_label.depth):
        for y in range(image_with_label.height):
            for x in range(image_with_label.width):
                if image_with_label.label_3d.image_3d[z,y,x] != 0:
                    image_3d[z,y,x] = image_with_label.image3d.image_3d[z,y,x]
                else:
                    continue
        print('have process {} slices'.format(z))
    save_image_3d(image_3d, image_save_root=image_path_save)

def enhance(ratio = 10., i = 30):
    image_root = '/home/li-qiufu/PycharmProjects/MyDataBase/0_HUST_DATA/2_cuted/' + 'N0' + str(i) + '/image'
    image_root_1 = '/home/li-qiufu/PycharmProjects/MyDataBase/0_HUST_DATA/2_cuted/' + 'N0' + str(i) + '/image_en'

    image_3d = load_image_3d(image_root=image_root)
    shape_ = [image_3d.shape[i] for i in range(len(image_3d.shape)) if image_3d.shape[i] != 1]
    image_3d = image_3d.reshape(shape_)

    image_3d = enhance_image_3d(image_3d=image_3d, ratio=ratio)
    save_image_3d(image_3d, image_save_root=image_root_1)

def main6(i = 10, r_offset = 10, No = 0):
    root = '/home/li-qiufu/PycharmProjects/MyDataBase/DataBase_5'
    source_root = os.path.join(root, str(i).zfill(6))

    image_path = os.path.join(source_root, 'image_' + str(No))
    file_swc = os.path.join(source_root, str(i).zfill(6) + '_' + str(No) +'.swc')
    image_path_save =  os.path.join(source_root, 'image_' + str(No+1))

    erasing(image_path = image_path, file_swc = file_swc, image_path_save = image_path_save, r_offset=r_offset)

def decode():
    root = '/home/li-qiufu/PycharmProjects/MyDataBase/DataBase_5/000031'
    file_name = os.path.join(root, '000031.v3draw')
    image_path = os.path.join(root, 'image_9')
    decode_image_v3draw(file_name=file_name, image_root_save=image_path)

def copy_from_to():
    root = '/home/li-qiufu/PycharmProjects/MyDataBase'
    from_path = os.path.join(root, 'DataBase_3')
    to_path = os.path.join(root, 'DataBase_4')
    according_path = os.path.join(root, 'DataBase_5')

    from_list = os.listdir(from_path)
    according_list = os.listdir(according_path)
    from_list.sort()
    for sub_name in from_list:
        if sub_name in according_list:
            continue
        sub_path_source = os.path.join(from_path, sub_name)
        os.system('cp -r' + ' ' + sub_path_source + ' ' + to_path)


if __name__ == '__main__':
    #main6(r_offset = 3, i = 87, No = 0)
    #enhance(ratio = 5, i = 85)
    #decode()
    #copy_from_to()
    """
    image_path_0 = '/home/li-qiufu/PycharmProjects/MyDataBase/0_HUST_DATA/2_cuted/N075/image_en'
    image_path_1 = '/home/li-qiufu/PycharmProjects/MyDataBase/0_HUST_DATA/2_cuted/N056/image_er_2'
    image_path_2 = '/home/li-qiufu/PycharmProjects/MyDataBase/0_HUST_DATA/2_cuted/N056/image_clear'
    image_0 = load_image_3d(image_path_0)
    image_1 = load_image_3d(image_path_1)
    image = image_0 - image_1
    save_image_3d(image, image_path_2)
    """

    image_path = '/home/li-qiufu/PycharmProjects/MyDataBase/0_HUST_DATA/4_image_label/HUST_DATA/angle_0/N075/image_er_1'
    file_swc = '/home/li-qiufu/PycharmProjects/MyDataBase/0_HUST_DATA/4_image_label/HUST_DATA/angle_0/N075/N075_noise_1.swc'
    image_path_save = '/home/li-qiufu/PycharmProjects/MyDataBase/0_HUST_DATA/4_image_label/HUST_DATA/angle_0/N075/image_er_2'
    erasing(image_path, file_swc, image_path_save, r_offset = 10)
