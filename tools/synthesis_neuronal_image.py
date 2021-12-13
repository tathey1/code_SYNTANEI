from tools.image_fusion_in_spatial_domain import Image3D_PATH, ImageWithLabel, NeuronImageWithLabel
from tools.image_fusion_in_spatial_domain import NeuronNodeList_SWC, NeuronNodeList
import numpy as np
from numpy.random import random
import copy
import os
import math
import random

class NeuronImageWithLabel_Noise(NeuronImageWithLabel):
    """
    伴随噪声标签的神经元数据类型
    """
    def __init__(self, image_path, file_swc, file_swc_noise, label = False):
        super(NeuronImageWithLabel_Noise, self).__init__(image_path = image_path, file_swc = file_swc, label = label)
        self.file_swc = file_swc
        self.image_path = image_path
        self.file_swc_noise = self._file_swc_noise(file_swc_noise)
        self.neuronnodelist_noise = NeuronNodeList_SWC(file_swc=self.file_swc_noise,
                                                       height=self.height,
                                                       width=self.width,
                                                       depth=self.depth)

    def _file_swc_noise(self, file_swc_noise):
        """
        合成噪声文件名，并生成相应的文件
        :return:
        """
        if not os.path.isfile(file_swc_noise):
            os.mknod(file_swc_noise)
        return file_swc_noise

    def add_noise_file_swc(self, noise_with_label, coor = (0,0,0), r_offset = 2):
        """
        将 image_with_label_noise 作为噪声添加到 image_with_label 的相应位置
        :param image_with_label: 被处理的神经元图像
        :param image_with_label_noise: 作为噪声添加的神经元数据
        :param coor: X(width),Y(height),Z(depth), 融合的起始坐标
        :param r_offset: 添加融合时候，噪声神经元的节点半径扩张值
        :return:
        """
        assert isinstance(noise_with_label, ImageWithLabel)
        assert noise_with_label.depth + coor[2] <= self.depth
        assert noise_with_label.height + coor[1] <= self.height
        assert noise_with_label.width + coor[0] <= self.width
        image_with_label_noise = copy.deepcopy(noise_with_label)
        image_with_label_noise = image_with_label_noise.translate(coor = coor)
        node_list_temp = NeuronNodeList()
        for key in image_with_label_noise.neuronnodelist.keys():
            node = copy.deepcopy(image_with_label_noise.neuronnodelist._neuron_node_list[key])
            radius = node.radius
            node.radius = r_offset + node.radius
            d0, d1, d2, d3 = self.neuronnodelist.distance_with_neuronnode(node)
            node.radius = radius
            node.child_id = list()
            if d1 == 0:
                node_list_temp.add_neuron_node(neuron_node=node)
            elif d2 >= 0.75 or d3 >= 0.75:
                node.radius = 0
                node_list_temp.add_neuron_node(neuron_node = node)
            else:
                node.radius = node.radius * (1 - d3) / 2
                node_list_temp.add_neuron_node(neuron_node=node)
        self.neuronnodelist_noise.concatenate(node_list_temp)
        node_list_temp.refresh_childID()
        image_with_label_noise.neuronnodelist.refresh_childID()
        node_list_temp.set_shape(image_with_label_noise.shape())
        keys = list(node_list_temp._neuron_node_list.keys())
        while keys != []:
            key = keys[0]
            while key:
                neuron_node = node_list_temp._neuron_node_list[key]
                if neuron_node.processed == 1 and neuron_node.child_id == []:
                    keys.remove(key)
                    break
                elif neuron_node.processed == 1 and neuron_node.child_id != []:
                    key = node_list_temp._neuron_node_list[key].child_id.pop(0)
                    continue
                elif neuron_node.processed == 0:
                    r_reset = r_offset + neuron_node.radius
                    points = neuron_node.get_around_points(r_reset = r_reset, shape = image_with_label_noise.shape())
                    for point in points:
                        self.image3d.image_3d[point] = image_with_label_noise.image3d.image_3d[point]
                        node_list_temp._neuron_node_list[key].processed = 1
                    if neuron_node.child_id != []:
                        neuron_node_child = node_list_temp._neuron_node_list[neuron_node.child_id[0]]
                        node_list_temp._neuron_node_list[key].child_id = neuron_node.child_id
                        points = neuron_node.get_connect_points(neuron_node_child,
                                                                shape = image_with_label_noise.shape(),
                                                                r_offset = r_offset)
                        for point in points:
                            self.image3d.image_3d[point] = image_with_label_noise.image3d.image_3d[point]
                        key = neuron_node.child_id[0]
                        neuron_node.child_id.pop(0)
                    else:
                        keys.remove(key)
                        break
        image_with_label_noise.neuronnodelist.refresh_childID()
        node_list_temp.refresh_childID()
        return self


class Add_Noise():
    """
    描述添加噪声过程的数据类型，确保噪声能合理的添加到神经元图像数据中
    """
    def __init__(self, image_with_label, image_with_label_noise, r_offset = 2, num = 1, info_file = None):
        """
        :param image_with_label: 神经元图像数据
        :param image_with_label_noise: 噪声图像数据
        :param num: 噪声添加次数
        """
        assert isinstance(image_with_label, NeuronImageWithLabel_Noise)
        assert isinstance(image_with_label_noise, NeuronImageWithLabel)
        self.image_with_label = image_with_label
        self.image_with_label_noise = image_with_label_noise
        self.r_offset = r_offset
        self.num = num
        self.noise_root_id = self._noise_root_id()
        self.noise_label = self._noise_label()
        self.info_write = False if info_file == None else True
        self.info_file = info_file if self.info_write == True else None
        self.resize_noise()

    def _noise_root_id(self):
        return list(self.image_with_label_noise.neuronnodelist.keys())[0]

    def _noise_label(self):
        """
        :return: 返回噪声图像的尺寸最大的轴
        """
        maximum = max(self.image_with_label_noise.depth,
                      self.image_with_label_noise.height,
                      self.image_with_label_noise.width)
        if self.image_with_label_noise.width == maximum:
            return 'x'
        if self.image_with_label_noise.height == maximum:
            return 'y'
        if self.image_with_label_noise.depth == maximum:
            return 'z'

    def resize_noise(self):
        """
        将噪声图像做尺度变换，使得变换后噪声图像数据能够合适的添加到神经元图像中
        :return:
        """
        if self.image_with_label_noise.depth <= 2 * self.image_with_label.depth / 3 and \
            self.image_with_label_noise.height <= 2 * self.image_with_label.height / 3 and \
            self.image_with_label_noise.width <= 2 * self.image_with_label.width / 3:
            shape_new = self.image_with_label_noise.shape()
            text = ' ' * 8 + 'resize to {}'.format(shape_new) + '\n'
            print(' ' * 8 + 'resize to {}'.format(shape_new))
            return text
        ratio = max(self.image_with_label_noise.depth / self.image_with_label.depth,
                    self.image_with_label_noise.height / self.image_with_label.height,
                    self.image_with_label_noise.width / self.image_with_label.width)
        ratio *= (3 / 2)
        shape_new = (round(self.image_with_label_noise.depth / ratio),
                     round(self.image_with_label_noise.height / ratio),
                     round(self.image_with_label_noise.width / ratio))
        self.image_with_label_noise.resize(shape_new = shape_new)
        text = ' ' * 8 + 'resize to {}'.format(shape_new) + '\n'
        print(' ' * 8 + 'resize to {}'.format(shape_new))
        return text

    def rotate_noise(self):
        """
        将噪声图像数据进行随机旋转
        :return:
        """
        _, (angle_x, angle_y, angle_z) = self.get_coor()
        print(' ' * 8 + 'angle_x, _y, _z = ({}, {}, {})'.format(angle_x, angle_y, angle_z))
        text = ' ' * 8 + 'angle_x, _y, _z = ({}, {}, {})'.format(angle_x * 180 / math.pi,
                                                                 angle_y * 180 / math.pi,
                                                                 angle_z * 180 / math.pi) + '\n'
        if angle_x == 0 and angle_y == 0 and angle_z == 0:
            return text
        self.image_with_label_noise.rotate(angle_X = angle_x, angle_Y = angle_y, angle_Z = angle_z)
        return text

    def get_coor(self):
        """
        获取噪声图像的插入坐标
        :return:
        """
        angle_x, angle_y, angle_z = 0, 0, 0
        x_first_node = self.image_with_label_noise.neuronnodelist._neuron_node_list[self.noise_root_id].x
        y_first_node = self.image_with_label_noise.neuronnodelist._neuron_node_list[self.noise_root_id].y
        z_first_node = self.image_with_label_noise.neuronnodelist._neuron_node_list[self.noise_root_id].z
        if self.noise_label == 'x':
            coor_y = random.randint(0, self.image_with_label.height - self.image_with_label_noise.height - 1)
            coor_z = random.randint(0, self.image_with_label.depth - self.image_with_label_noise.depth - 1)
            if x_first_node < (self.image_with_label_noise.width / 2):
                coor_x = 0
            else:
                coor_x = self.image_with_label.width - self.image_with_label_noise.width
            angle_x = random.randint(-45,45) * math.pi / 180
        elif self.noise_label == 'y':
            coor_x = random.randint(0, self.image_with_label.width - self.image_with_label_noise.width - 1)
            coor_z = random.randint(0, self.image_with_label.depth - self.image_with_label_noise.depth - 1)
            if y_first_node < (self.image_with_label_noise.height / 2):
                coor_y = 0
            else:
                coor_y = self.image_with_label.height - self.image_with_label_noise.height
            angle_y = random.randint(-45, 45) * math.pi / 180
        elif self.noise_label == 'z':
            coor_x = random.randint(0, self.image_with_label.width - self.image_with_label_noise.width - 1)
            coor_y = random.randint(0, self.image_with_label.height - self.image_with_label_noise.height - 1)
            if z_first_node < (self.image_with_label_noise.depth / 2):
                coor_z = 0
            else:
                coor_z = self.image_with_label.depth - self.image_with_label_noise.depth
            angle_z = random.randint(5, 180) * math.pi / 180
        else:
            raise ValueError
        return (coor_x, coor_y, coor_z), (angle_x, angle_y, angle_z)

    def add_noise(self):
        """
        向神经元图像中添加 num 次噪声数据
        :return:
        """
        print(' ' * 4 + 'rotating ...')
        text = ' ' * 4 + 'rotating ...\n'
        text += self.rotate_noise()
        print(' ' * 4 + 'resizing ...')
        text += ' ' * 4 + 'resizing ...\n'
        text += self.resize_noise()
        self.image_with_label_noise.cut_whole()
        for i in range(self.num):
            print(' ' * 4 + 'add time - {}'.format(i))
            text += ' ' * 4 + 'add time - {}'.format(i) + '\n'
            coor, _ = self.get_coor()
            print(' ' * 8 + 'adding ... coor(x,y,z) = ({},{},{})'.format(coor[0],coor[1],coor[2]))
            text += ' ' * 8 + 'adding ... coor(x,y,z) = ({},{},{})'.format(coor[0],coor[1],coor[2]) + '\n'
            self.image_with_label.add_noise_file_swc(noise_with_label = self.image_with_label_noise,
                                                     coor = coor,
                                                     r_offset = self.r_offset)
        if self.info_write:
            self.info_file.write(text)


class Add_Mutil_Noise():
    """
    往一个神经元图像中添加多个噪声数据的类型
    """
    def __init__(self, image_with_label, root_noise = None, r_offset = 2, num_max = 5, num_min = 2, number = 20, info_file = None):
        """
        :param image_with_label: 神经元图像数据
        :param root_noise: 噪声数据保存路径
        :param r_offset: 噪声半径增量值
        :param num_max: 每个噪声添加的最大次数（在当前神经元数据中）
        :param num_min: 每个噪声添加的最小次数
        :param number: 添加的噪声个数
        """
        self.image_with_label = image_with_label
        self.root_noise = '/home/li-qiufu/PycharmProjects/MyDataBase/Neuron_Branch' if root_noise == None else root_noise
        self.r_offset = r_offset
        self.num_max = num_max
        self.num_min = num_min
        self.number = number
        self.branch_name_list = self._branch_name_list()
        self.info_write = False if info_file == None else True
        self.info_file = info_file if self.info_write == True else None

    def _branch_name_list(self):
        """
        :return: 随机罗列出 self.root_noise 中的 self.number 个噪声数据名称
        """
        #print(len(os.listdir(self.root_noise)))
        branch_list = random.sample(range(0, 297), self.number)
        #branch_list = [0,116,80,273,189,293,186,244,201,269]
        return list('branch_' + str(x).zfill(6) for x in branch_list)

    def add_noise(self):
        """
        将已经确定的 self.number 个噪声数据添加到神经元图像中
        :return:
        """
        for index, branch_name in enumerate(self.branch_name_list):
            print('\n  {} / {} -- '.format(index, self.number) + branch_name)
            if self.info_write:
                self.info_file.write('\n  {} / {} -- '.format(index, self.number) + branch_name + '\n')
            image_path_noise = os.path.join(self.root_noise, branch_name, 'image')
            file_swc_noise = os.path.join(self.root_noise, branch_name, branch_name + '.swc')
            image_with_label_noise = NeuronImageWithLabel(image_path = image_path_noise,
                                                          file_swc = file_swc_noise,
                                                          label = False)
            num = random.randint(self.num_min, self.num_max)
            add_noise_action = Add_Noise(image_with_label = self.image_with_label,
                                         image_with_label_noise = image_with_label_noise,
                                         r_offset = 3, num = num, info_file = self.info_file)
            add_noise_action.add_noise()


def add_noise(source, target, number = 15):
    if not os.path.isdir(target):
        os.mkdir(target)
    sub_path_list = os.listdir(source)
    sub_path_list.sort()
    for sub_path in sub_path_list:
        root_source = os.path.join(source, sub_path)
        if not os.path.isdir(root_source):
            continue
        root_target = os.path.join(target, sub_path)

        if not os.path.isdir(root_target):
            os.mkdir(root_target)
        info = open(os.path.join(root_target, 'noise.info'), 'w')
        object_list = os.listdir(root_source)
        object_list.sort()
        for object_ in object_list:
            print('\n\n\n' + os.path.join(sub_path, object_))
            info.write('\n\n\n' + os.path.join(sub_path, object_) + '\n')
            if not os.path.isdir(os.path.join(root_source, object_)):
                continue
            image_path = os.path.join(root_source, object_, 'image')
            file_swc = os.path.join(root_source, object_, object_ + '_0.swc')
            if not os.path.isdir(os.path.join(root_target, object_)):
                os.mkdir(os.path.join(root_target, object_))
            image_path_save = os.path.join(root_target, object_, 'image')
            file_swc_save = os.path.join(root_target, object_, object_ + '.swc')
            file_swc_noise_save = os.path.join(root_target, object_, object_ + '_noise.swc')
            image_with_label = NeuronImageWithLabel_Noise(image_path = image_path,
                                                          file_swc = file_swc,
                                                          file_swc_noise = file_swc_noise_save)
            if image_with_label.map_size() <= 60000:
                number_ = round(number * 0.7)
            elif image_with_label.map_size() <= 80000:
                number_ = round(number * 0.85)
            else:
                number_ = round(number)
            add_noise_action = Add_Mutil_Noise(image_with_label = image_with_label,
                                               num_min = 1,
                                               number = number_,
                                               info_file = info)
            add_noise_action.add_noise()
            image_with_label.save(image_save_root = image_path_save, saved_file_name = file_swc_save)
            image_with_label.neuronnodelist_noise.save(saved_file_name = file_swc_noise_save)

            path_source = os.path.join(root_source, object_)
            path_target = os.path.join(root_target, object_)
            swc_noise_list = os.listdir(path_source)
            swc_noise_list.sort()
            for swc_noise in swc_noise_list:
                if '_noise_' not in swc_noise:
                    continue
                file_swc_noise = os.path.join(path_source, swc_noise)
                file_swc_noise_save = os.path.join(path_target, swc_noise)
                neuron_node_list = NeuronNodeList_SWC(file_swc = file_swc_noise,
                                                      depth = image_with_label.depth,
                                                      height = image_with_label.height,
                                                      width = image_with_label.width)
                neuron_node_list.save(saved_file_name = file_swc_noise_save)
            info.flush()
        info.close()

def translate(target):
    source = '/home/li-qiufu/PycharmProjects/MyDataBase/not_center_enhance/0_orginal'
    source_swc = '/home/li-qiufu/PycharmProjects/MyDataBase/not_center/0_orginal'
    neuron_list = os.listdir(source)
    for neuron in neuron_list:
        neuron_path = os.path.join(source, neuron)
        image_path = os.path.join(neuron_path, 'image')
        image_path_save = os.path.join(target, neuron, 'image')
        swc_file_save = os.path.join(target, neuron, neuron + '_0.swc')
        swc_file = os.path.join(source_swc, neuron, neuron + '_0.swc')
        nil = NeuronImageWithLabel(image_path = image_path, file_swc = swc_file, resolution = [1,1,1])
        nil.translate(coor = (round(nil.width / 2), round(nil.height / 2), 0)).save(image_save_root = image_path_save, saved_file_name = swc_file_save)

def add_noise_1(target, number = 15):
    source = '/home/li-qiufu/PycharmProjects/MyDataBase/not_center_enhance/1_tran'
    info = open(os.path.join(source, 'noise.info'), 'w')
    object_list = os.listdir(source)
    object_list.sort()
    for object_ in object_list:
        print('\n\n\n' + os.path.join(source, object_))
        info.write('\n\n\n' + os.path.join(source, object_) + '\n')
        image_path = os.path.join(source, object_, 'image')
        file_swc = os.path.join(source, object_, object_ + '_0.swc')
        if not os.path.isdir(os.path.join(target, object_)):
            os.mkdir(os.path.join(target, object_))
        image_path_save = os.path.join(target, object_, 'image')
        file_swc_save = os.path.join(target, object_, object_ + '.swc')
        file_swc_noise_save = os.path.join(target, object_, object_ + '_noise.swc')
        image_with_label = NeuronImageWithLabel_Noise(image_path = image_path,
                                                      file_swc = file_swc,
                                                      file_swc_noise = file_swc_noise_save)
        add_noise_action = Add_Mutil_Noise(image_with_label = image_with_label,
                                           num_min = 1,
                                           number = number,
                                           info_file = info)
        add_noise_action.add_noise()
        image_with_label.save(image_save_root = image_path_save, saved_file_name = file_swc_save)
        image_with_label.neuronnodelist_noise.save(saved_file_name = file_swc_noise_save)

        path_source = os.path.join(source, object_)
        path_target = os.path.join(target, object_)
        swc_noise_list = os.listdir(path_source)
        swc_noise_list.sort()
        for swc_noise in swc_noise_list:
            if '_noise_' not in swc_noise:
                continue
            file_swc_noise = os.path.join(path_source, swc_noise)
            file_swc_noise_save = os.path.join(path_target, swc_noise)
            neuron_node_list = NeuronNodeList_SWC(file_swc = file_swc_noise,
                                                  depth = image_with_label.depth,
                                                  height = image_with_label.height,
                                                  width = image_with_label.width)
            neuron_node_list.save(saved_file_name = file_swc_noise_save)
        info.flush()
    info.close()

from image_3D_io import *
def oo(source):
    neuron_list = os.listdir(source)
    for neuron in neuron_list:
        image_path = os.path.join(source, neuron, 'image')
        swc_file = os.path.join(source, neuron, neuron + '.swc')
        swc_file_noise = os.path.join(source, neuron, neuron + '_noise.swc')
        nil_0 = NeuronImageWithLabel(image_path = image_path, file_swc = swc_file, label = True)
        nil_1 = NeuronImageWithLabel(image_path = image_path, file_swc = swc_file_noise, label = True)
        nil_1.label_3d.image_3d[nil_1.label_3d.image_3d == 1] = 2
        label_3d = nil_0.label_3d.image_3d + nil_1.label_3d.image_3d
        label_3d[label_3d == 3] = 1
        save_image_3d(label_3d, os.path.join(source, neuron, 'label'))

def xx(source, target):
    image_path = os.path.join(source, 'image')
    label_path = os.path.join(source, 'label')
    image_3d = Image3D_PATH(image_path = image_path)
    label_3d = Image3D_PATH(image_path = label_path)
    image_3d.image_3d[label_3d.image_3d != 1] = 0
    image_3d.save(image_save_root = os.path.join(target, 'image'))

def resize(source, target):
    neuron_list = os.listdir(source)
    for neuron in neuron_list:
        print('processing {}', os.path.join(source, neuron))
        image_path = os.path.join(source, neuron, 'image')
        swc_file = os.path.join(source, neuron, neuron + '.swc')
        swc_file_noise = os.path.join(source, neuron, neuron + '_noise.swc')
        nil_0 = NeuronImageWithLabel(image_path = image_path, file_swc = swc_file, label = False)
        nil_1 = NeuronImageWithLabel(image_path = image_path, file_swc = swc_file_noise, label = False)
        nil_0.resize(shape_new = (32,128,128)).save(image_save_root = os.path.join(target, neuron, 'image'),
                                                    saved_file_name = os.path.join(target, neuron, neuron + '.swc'))
        nil_1.resize(shape_new = (32,128,128)).save(saved_file_name = os.path.join(target, neuron, neuron + '_noise.swc'))
        label_path = os.path.join(source, neuron, 'label')
        image_3d = Image3D_PATH(image_path = label_path)
        image_3d.resize(shape_new = (32,128,128)).save(image_save_root = os.path.join(target, neuron, 'label'))


if __name__ == '__main__':
    """
    info = open('/home/li-qiufu/PycharmProjects/MyDataBase/test_sample/info', 'w')
    neuron_image = '/home/li-qiufu/PycharmProjects/MyDataBase/test_sample/image_neuron/000000/image'
    neuron_file_swc = '/home/li-qiufu/PycharmProjects/MyDataBase/test_sample/image_neuron/000000/000000_0.swc'
    neuron_file_swc_noise = '/home/li-qiufu/PycharmProjects/MyDataBase/test_sample/image_neuron/000000/000000_noise.swc'
    neuron_image_save = '/home/li-qiufu/PycharmProjects/MyDataBase/test_sample/image_neuron/000000_noise/image'
    neuron_image_1 = '/home/li-qiufu/PycharmProjects/MyDataBase/test_sample/image_neuron/000000/image'
    neuron_file_swc_1 = '/home/li-qiufu/PycharmProjects/MyDataBase/test_sample/image_neuron/000006/000006_0.swc'
    neuron_file_swc_noise_1 = '/home/li-qiufu/PycharmProjects/MyDataBase/test_sample/image_neuron/000006/000006_noise.swc'
    neuron_image_save_1 = '/home/li-qiufu/PycharmProjects/MyDataBase/test_sample/image_neuron/000006_noise/image'

    image_with_label = NeuronImageWithLabel_Noise(image_path = neuron_image,
                                                  file_swc = neuron_file_swc,
                                                  file_swc_noise = neuron_file_swc_noise)
    image_with_label_1 = NeuronImageWithLabel_Noise(image_path = neuron_image_1,
                                                  file_swc = neuron_file_swc_1,
                                                  file_swc_noise = neuron_file_swc_noise_1)
    root_noise = '/home/li-qiufu/PycharmProjects/MyDataBase/Neuron_Branch'

    add_multi_noise = Add_Mutil_Noise(image_with_label = image_with_label,
                                      root_noise = root_noise, number = 10, info_file = info)
    add_multi_noise.add_noise()
    image_with_label.save(image_save_root = neuron_image_save)
    image_with_label.neuronnodelist_noise.save(saved_file_name = image_with_label.file_swc_noise)

    add_multi_noise_1 = Add_Mutil_Noise(image_with_label = image_with_label_1,
                                        root_noise = root_noise, number = 10, info_file = info)
    add_multi_noise_1.add_noise()
    image_with_label_1.save(image_save_root = neuron_image_save_1)
    image_with_label_1.neuronnodelist_noise.save(saved_file_name = image_with_label_1.file_swc_noise)
    info.close()
    #add_noise = Add_Noise(image_with_label = image_with_label, image_with_label_noise = image_with_label_noise)
    #add_noise.add_noise()
    
    source = '/home/li-qiufu/PycharmProjects/MyDataBase/DataBase_15'
    target_1 = '/home/li-qiufu/PycharmProjects/MyDataBase/DataBase_19'
    target_2 = '/home/li-qiufu/PycharmProjects/MyDataBase/DataBase_20'
    target_3 = '/home/li-qiufu/PycharmProjects/MyDataBase/DataBase_21'
    add_noise(source = source, target = target_1, number = 10)
    add_noise(source = source, target = target_2, number = 15)
    add_noise(source = source, target = target_3, number = 18)
    
    target = '/home/li-qiufu/PycharmProjects/MyDataBase/not_center/tran'
    translate(target)
    
    target = '/home/li-qiufu/PycharmProjects/MyDataBase/not_center/tangled'
    add_noise_1(target, number = 15)
    
    source = '/home/li-qiufu/PycharmProjects/MyDataBase/not_center/2_tangled'
    target = '/home/li-qiufu/PycharmProjects/MyDataBase/not_center/3_resized'
    resize(source, target)
    """

    source = '/home/li-qiufu/PycharmProjects/MyDataBase/not_center_enhance/2_tangled/000089'
    target = '/home/li-qiufu/PycharmProjects/MyDataBase/not_center_enhance/5_target/000089'

    xx(source,target)