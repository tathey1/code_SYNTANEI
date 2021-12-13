"""
这个脚本将多个噪声swc文件合成一个
"""
import os
import math
from tools.image_fusion_in_spatial_domain import *

def concatenate_noise(root):
    """
    目录 root 中含有多个噪声swc文件，
    这个函数将多个噪声swc文件合成一个
    :param root:
    :return:
    """
    filename_list = os.listdir(root)
    image_path = os.path.join(root, 'image')
    image3d_path = Image3D_PATH(image_path = image_path)
    noise_neuron_node_list = NeuronNodeList()
    noise_neuron_node_list.set_shape(shape = image3d_path.shape())
    for filename in filename_list:
        if not '_noise_' in filename:
            continue
        file_swc_name = os.path.join(root, filename)
        neuron_node_list_swc = NeuronNodeList_SWC(file_swc = file_swc_name,
                                                  depth = image3d_path.depth,
                                                  height = image3d_path.height,
                                                  width = image3d_path.width)
        noise_neuron_node_list.concatenate(neuron_node_list_swc)
    noise_neuron_node_list.save(saved_file_name = os.path.join(root, 'noise.swc'))

def generate_label_3d_noise(root, name = 'N041'):
    """
    生成图像的矩阵标签
    目标神经元生成1，干扰纤维生成2，其他位置生成0
    :param root:
    :return:
    """
    image_path = os.path.join(root, 'image')
    file_swc_name = os.path.join(root, name + '.swc')
    file_swc_name_noise = os.path.join(root, 'noise.swc')
    image3d_path = Image3D_PATH(image_path = image_path)
    neuron_node_list = NeuronNodeList_SWC(file_swc = file_swc_name,
                                                  depth = image3d_path.depth,
                                                  height = image3d_path.height,
                                                  width = image3d_path.width,
                                                resolution = [1,1,1/0.32])
    neuron_node_list_noise = NeuronNodeList_SWC(file_swc = file_swc_name_noise,
                                                  depth = image3d_path.depth,
                                                  height = image3d_path.height,
                                                  width = image3d_path.width,
                                                resolution = [1,1,1/0.32])
    neuron_node_list.generate_label_3d(label_mark = 1)
    neuron_node_list_noise.generate_label_3d(label_mark = 2)
    label_3d = neuron_node_list.label_3d.image_3d + neuron_node_list_noise.label_3d.image_3d
    save_image_3d(label_3d, image_save_root = os.path.join(root, 'label'))

def rotate(root, save_root, angle = 60):
    """
    将图像、swc文件、标签矩阵绕 z 轴进行旋转 angle 角度
    :param root:
    :return:
    """
    neuron_name_list = os.listdir(root)
    neuron_name_list.sort()
    for neuron_name in neuron_name_list:
        neuron_root = os.path.join(root, neuron_name)
        print('we are processing {} -- {}'.format(neuron_root, angle))
        neuron_save_root = os.path.join(save_root, neuron_name)

        image_path = os.path.join(neuron_root, 'image')
        swc_file = os.path.join(neuron_root, neuron_name + '.swc')
        swc_noise_file = os.path.join(neuron_root, 'noise.swc')
        label_path = os.path.join(neuron_root, 'label')

        image = Image3D_PATH(image_path = image_path)
        label = Image3D_PATH(image_path = label_path)
        neuron_node_list = NeuronNodeList_SWC(file_swc = swc_file,
                                              depth = image.depth,
                                              height = image.height,
                                              width = image.width)
        neuron_node_noise_list = NeuronNodeList_SWC(file_swc = swc_noise_file,
                                              depth = image.depth,
                                              height = image.height,
                                              width = image.width)
        image = image.rotate(angle_Z = angle * math.pi / 180)
        label = label.rotate(angle_Z = angle * math.pi / 180)
        neuron_node_list = neuron_node_list.rotate(angle_Z = angle * math.pi / 180)
        neuron_node_noise_list = neuron_node_noise_list.rotate(angle_Z = angle * math.pi / 180)
        image.save(image_save_root = os.path.join(neuron_save_root, 'image'))
        label.save(image_save_root = os.path.join(neuron_save_root, 'label'))
        neuron_node_list.save(saved_file_name = os.path.join(neuron_save_root, neuron_name + '.swc'))
        neuron_node_noise_list.save(saved_file_name = os.path.join(neuron_save_root, 'noise.swc'))


def resize(root, save_root):
    neuron_name_list = os.listdir(root)
    neuron_name_list.sort()
    for neuron_name in neuron_name_list:
        neuron_root = os.path.join(root, neuron_name)
        print('we are resizing {} \n ---- {}'.format(neuron_root, root))
        neuron_save_root = os.path.join(save_root, neuron_name)
        image_path = os.path.join(neuron_root, 'image')
        label_path = os.path.join(neuron_root, 'label')
        image = Image3D_PATH(image_path = image_path)
        label = Image3D_PATH(image_path = label_path)
        image = image.resize(shape_new = (32,128,128))
        label = label.resize(shape_new = (32,128,128))
        image.save(image_save_root = os.path.join(neuron_save_root, 'image'))
        label.save(image_save_root = os.path.join(neuron_save_root, 'label'))


if __name__ == '__main__':
    root = '/home/li-qiufu/PycharmProjects/MyDataBase/0_HUST_DATA/4_image_label/angle_300'
    save_root = '/home/li-qiufu/PycharmProjects/MyDataBase/0_HUST_DATA/5_resized/angle_300'
    resize(root = root, save_root = save_root)
