"""
这个脚本描述神经元图像数据，以及由此图像数据构成的适用于 pytorch 的数据集合类型
"""

from tools.image_fusion_in_spatial_domain import Image3D_PATH
from tools.image_fusion_in_spatial_domain import NeuronNodeList_SWC
from tools.image_3D_io import save_image_3d, load_image_3d
import os
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset


class Neuron_Data():
    """
    对神经元图像数据进行描述
    """
    def __init__(self, data_path):
        self.data_path = data_path
        self.image = None
        self.label = None
        self._image_label_swc()
        self._check()
        self._data_label_noise()

    def __repr__(self):
        """
        对这个神经元数据进行描述
        :return:
        """
        text = 'A neuron DATA generated from "{}"'.format(self.data_path)
        return text

    def _image_label_swc(self):
        """
        生成相关属性并赋值，这里要确保相应的 swc 标签文件的文件名与神经元名称一致，而 swc 噪声文件名中含有 'noise'
        :return:
        """
        assert os.path.isdir(self.data_path)
        self.image_path = os.path.join(self.data_path, 'image')
        assert os.path.isdir(self.image_path)
        self.label_path = os.path.join(self.data_path, 'label')
        self.data_path = self.data_path.rstrip(os.path.sep)
        _, self.name = os.path.split(self.data_path)
        self.file_swc = os.path.join(self.data_path, self.name + '.swc')
        file_swc_noise_list = os.listdir(self.data_path)
        self.file_swc_noise_list = [os.path.join(self.data_path, file_swc_noise)
                                    for file_swc_noise in file_swc_noise_list if 'noise' in file_swc_noise]

    def _check(self):
        """
        检查标签数据是否存在
        :return:
        """
        if not os.path.isdir(self.label_path):
            os.mkdir(self.label_path)
            self.label_exist = False
        elif len(os.listdir(self.label_path)) != len(os.listdir(self.image_path)):
            self.label_exist = False
        else:
            self.label_exist = True

    def _data_label_noise(self):
        """
        生成标签矩阵，并将相应的数据标签等转换为可处理对象类型
        :return:
        """
        self.image = Image3D_PATH(image_path = self.image_path)
        self.depth = self.image.depth
        self.height = self.image.height
        self.width = self.image.width
        self.depth_orignal = self.image.depth
        self.height_orignal = self.image.height
        self.width_orignal = self.image.width
        if self.label_exist:
            self.label = Image3D_PATH(image_path = self.label_path)
            assert self.label.height == self.image.height
            assert self.label.width == self.image.width
        else:
            self.neuron_node_label = NeuronNodeList_SWC(file_swc = self.file_swc,
                                                        depth = self.depth,
                                                        height = self.height,
                                                        width = self.width,
                                                        label = True)
            self.neuron_node_noise_list = [NeuronNodeList_SWC(file_swc = file_swc,
                                                              depth = self.depth,
                                                              height = self.height,
                                                              width = self.width,
                                                              label = False)
                                           for file_swc in self.file_swc_noise_list]
            self._generate_label()

    def _generate_label(self, r_offset = 2, noise_mark = 2):
        """
        生成标签数据
        :return:
        """
        if self.label_exist:
            return
        else:
            self.label = self.neuron_node_label.label_3d
            for neuron_node_noise in self.neuron_node_noise_list:
                keys = list(neuron_node_noise._neuron_node_list.keys())
                while keys != []:
                    key = keys[0]
                    while key:
                        neuron_node = neuron_node_noise._neuron_node_list[key]
                        if neuron_node.processed == 1 and neuron_node.child_id == []:
                            keys.remove(key)
                            break
                        elif neuron_node.processed == 1 and neuron_node.child_id != []:
                            key = neuron_node_noise._neuron_node_list[key].child_id.pop(0)
                            continue
                        elif neuron_node.processed == 0:
                            r_reset = r_offset + neuron_node.radius
                            points = neuron_node.get_around_points(r_reset = r_reset, shape = self.shape())
                            for point in points:
                                if self.label.image_3d[point] == 0:
                                    self.label.image_3d[point] = noise_mark
                                neuron_node_noise._neuron_node_list[key].processed = 1
                            if neuron_node.child_id != []:
                                neuron_node_child = neuron_node_noise._neuron_node_list[neuron_node.child_id[0]]
                                neuron_node_noise._neuron_node_list[key].child_id = neuron_node.child_id
                                points = neuron_node.get_connect_points(neuron_node_child,
                                                                        shape = self.shape(),
                                                                        r_offset = r_offset)
                                for point in points:
                                    if self.label.image_3d[point] == 0:
                                        self.label.image_3d[point] = noise_mark
                                key = neuron_node.child_id[0]
                                neuron_node.child_id.pop(0)
                                continue
                            else:
                                keys.remove(key)
                                break

    def shape(self):
        return self.depth, self.height, self.width

    def shape_orignal(self):
        return self.depth_orignal, self.height_orignal, self.width_orignal

    def save_label(self):
        """
        保存标签数据
        :return:
        """
        if self.label_exist:
            return
        save_image_3d(self.label.image_3d, image_save_root = self.label_path, dim = 0)

    def resize(self, shape_new):
        """
        将数据和标签3D矩阵进行变形
        :return:
        """
        if shape_new[0] == self.depth and shape_new[1] == self.height and shape_new[2] == self.width:
            return
        elif shape_new[0] == 0 or shape_new[1] == 0 or shape_new[2] == 0:
            return
        else:
            self.image.resize(shape_new = shape_new)
            self.label.resize(shape_new = shape_new)
            self.depth, self.height, self.width = shape_new


class Neuron_Data_Set(Dataset):
    """
    定义神经元数据集
    """
    def __init__(self, root, source, channel = 1, depth = 64, height = 256, width = 256):
        self.root = root
        self.source = source
        self.neuron_name_list = self._neuron_name_list()
        self.channel = channel
        self.depth = depth
        self.height = height
        self.width = width

    def _neuron_name_list(self):
        """
        生成完整的神经元数据路径名列表
        :return:
        """
        assert os.path.isfile(self.source)
        neuron_name_list = open(self.source).readlines()
        neuron_name_list = [line.strip() for line in neuron_name_list]
        neuron_name_list = [line for line in neuron_name_list if not line.startswith('#')]
        return [line for line in neuron_name_list if os.path.isdir(os.path.join(self.root, line))]

    def __getitem__(self, item):
        """
        返回数据的第 item 项
        :param item:
        :return:
        """
        data_path = os.path.join(self.root, self.neuron_name_list[item])
        image = load_image_3d(image_root = os.path.join(data_path, 'image'))
        label = load_image_3d(image_root = os.path.join(data_path, 'label'))
        return torch.Tensor(image).float(), torch.Tensor(label).long()

    def __len__(self):
        return len(self.neuron_name_list)

    def __repr__(self):
        """
        对当前数据集进行描述
        :return:
        """
        text = 'A neuron DATA SET generated from "{}"'.format(self.root)
        return text


if __name__ == '__main__':
    def main0(data_root):
        angles = ['angle_0', 'angle_60', 'angle_120', 'angle_180', 'angle_240', 'angle_300']
        for angle in angles:
            image_root = os.path.join(data_root, angle)
            object_list = os.listdir(image_root)
            object_list.sort()
            for object_ in object_list:
                data_path = os.path.join(image_root, object_)
                print('processing {}'.format(data_path))
                neuron_data = Neuron_Data(data_path = data_path)
                neuron_data.save_label()

    data_root_0 = '/home/li-qiufu/PycharmProjects/MyDataBase/neuron_data_1/DataBase_19'
    data_root_1 = '/home/li-qiufu/PycharmProjects/MyDataBase/neuron_data_1/DataBase_20'
    data_root_2 = '/home/li-qiufu/PycharmProjects/MyDataBase/neuron_data_1/DataBase_21'
    main0(data_root = data_root_0)
    main0(data_root = data_root_1)
    main0(data_root = data_root_2)
