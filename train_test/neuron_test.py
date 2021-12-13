"""
这个脚本对使用训练好的模型对测试数据进行测试
"""
import os
import torch
import numpy as np
from train_test.neuron_data import Neuron_Data, Neuron_Data_Set
from torch.utils.data.dataloader import DataLoader
from networks.neuron_net import *
from torch.autograd import Variable
from datetime import datetime
from torch.nn.parallel import DataParallel
from tools.image_fusion_in_spatial_domain import Image3D, Image3D_PATH
from copy import deepcopy
from train_test.neuron_args import ARGS
from torch.utils.data import Dataset

MODEL_NAME = '/home/liqiufu/neuron_pytorch_dl/weights_model/neuron_segmentation_U_80/neuron_segmentation_U_80_epoch_150.pkl'
NET = Neuron_Net_U_BN_V12

class Neuron_Data_Set_Test(Dataset):
    """
    用于测试的神经元图像数据集类型，它返回图像数据和图像名称，或者还包含其他属性
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
        data_path = os.path.join(self.root, self.neuron_name_list[item], 'image')
        neuron_data = Image3D_PATH(image_path = data_path)
        #print(data_path)
        neuron_data.resize(shape_new = (self.depth, self.height, self.width))
        image = neuron_data.image_3d.reshape((self.channel, neuron_data.depth, neuron_data.height, neuron_data.width))
        image = (image - 127.5) / 255
        name = self.neuron_name_list[item]
        return torch.Tensor(image).float(), name

    def __len__(self):
        return len(self.neuron_name_list)


class Neuron_Test():
    """
    使用训练好的神经元图像去噪模型对复杂背景的神经元图像进行分割处理，并保存分割结果
    """
    def __init__(self, root = None, source = None, batch = 8, channel = 1,
                 depth = ARGS['neuron_depth'],
                 height = ARGS['neuron_height'],
                 width = ARGS['neuron_width'],
                 root_saved = None):
        """
        :param root:    测试数据保存路径
        :param source:  测试数据文件名列表保存文件
        :param batch:   测试时候 batchsize 大小
        :param channel: 测试时候图像的通道数，不需要进行设置，使用默认值即可
        :param depth:   测试时候将数据的尺寸变换为(depth, height, width)大小
        :param height:  若(depth, height, width) = (0, 0, 0)，则表示不进行尺寸变换，此时需要设置 batch = 1
        :param width:   若图像的尺寸小于这里的默认值，则必须进行尺寸变换，否则测试无法正常进行
        :param root_saved:  测试结果保存路径
        """
        self.root = root
        self.root_saved = root_saved
        self.source = source
        self.batch = batch
        self.channel = channel
        self.depth = depth
        self.height = height
        self.width = width
        self.resize_flag = True
        self._check()
        self._data()
        self.net = NET()
        self.load_pretrained_model()#加载预训练模型
        self.current_index = 0      #记录已经处理了多少个吐下你给

    def _check(self):
        """
        检查该类型的某些属性值是否符合要求
        :return:
        """
        if self.depth == 0 or self.height == 0 or self.width == 0:
            self.resize_flag = False
            if self.batch != 1:
                raise ValueError('若(depth, height, width)中的某个值设置为0，表示不对图像立方体进行变换，此时必须将batch设置为 1')
        if not os.path.isdir(self.root):
            raise AttributeError('图像保存路径不存在')
        if not os.path.isfile(self.source):
            raise AttributeError('图像名列表保存文件不存在')
        if self.channel != 1:
            raise ValueError('channel 值必须设置为 1')

    def _data(self):
        """
        处理待测试数据的路径值
        :return:
        """
        self.neuron_name_list = []
        neuron_lines = open(self.source).readlines()
        neuron_lines = [line.strip() for line in neuron_lines]
        for line in neuron_lines:
            neuron_root = os.path.join(self.root, line)
            if not os.path.isdir(neuron_root):
                continue
            self.neuron_name_list.append(line)
        if len(self.neuron_name_list) == 0:
            raise AttributeError('图像路径中没有数据')
        self.neuron_set = Neuron_Data_Set_Test(root = self.root, source = self.source,
                                               depth = self.depth, height = self.height, width = self.width)
        self.data_loader = DataLoader(dataset = self.neuron_set, batch_size = self.batch,
                                      shuffle = False, num_workers = 4)

    def load_pretrained_model(self):
        """
        加载预训练的模型参数，有可能某些模型参数被修改或增加减少某些层，这些情况应被处理
        :return:
        """
        self.net = DataParallel(module = self.net.cuda(), device_ids = ARGS['gpu'], output_device = ARGS['out_gpu'])
        model_dict = self.net.state_dict()
        pretrained_dict = torch.load(MODEL_NAME, map_location = ARGS['gpu_map'])
        pretrained_dict = [(k, v) for k, v in pretrained_dict.items() if k in model_dict]
        model_dict.update(pretrained_dict)
        self.net.load_state_dict(model_dict)
        self.net.eval()

    def run(self):
        """
        对数据进行测试
        :return:
        """
        start = datetime.now()
        for index, (neuron_image, neuron_name) in enumerate(self.data_loader):
            start_1 = datetime.now()
            neuron_image = Variable(neuron_image, requires_grad = False).cuda()
            output = self.net.forward(neuron_image)
            #label_pridicted = output.data.cpu().numpy().argmax(axis = 1)
            _, label_pridicted = output.topk(1, dim = 1)
            print(label_pridicted.size())
            label_pridicted = torch.squeeze(label_pridicted,1)
            print(label_pridicted.size())
            label_pridicted = label_pridicted.cpu().numpy()
            print(label_pridicted.shape)
            self.save_pridicted_data(neuron_name, label_pridicted)
            stop_1 = datetime.now()
            print('    took {} hours'.format(stop_1 - start_1))
        stop = datetime.now()
        print('took {} hours totally'.format(stop - start))

    def save_pridicted_data(self, neuron_name_batch, label_pridicted):
        """
        保存预测结果
        :param neuron_name_batch: 当前批次的图像名列表
        :param neuron_size_batch: 当前批次图像的原始尺寸列表
        :param label_pridicted: 当前批次的预测标签
        :return:
        """
        for i in range(len(neuron_name_batch)):
            label_p = label_pridicted[i]
            pre_3d = Image3D()
            pre_3d.image_3d = label_p
            pre_3d.refresh_shape()

            neuron_name = neuron_name_batch[i]
            neuron_root_saved = os.path.join(self.root_saved, neuron_name)
            if not os.path.isdir(neuron_root_saved):
                os.makedirs(neuron_root_saved)
            pre_save_root_label = os.path.join(neuron_root_saved, 'label_pre')
            pre_3d.save(image_save_root = pre_save_root_label, dim = 0)


class Test_Post_Processing():
    """
    Neuron_Test 分割得到的神经元图像是标准大小的(32*128*128 或 64*256*256)，需将结果变换为图像原始大小
    或者还需要根据分割结果将神经元图像中的相应部分提取出来
    """
    def __init__(self, root, source, test_result_root):
        """
        :param root: 保存原始尺寸图像数据的根目录
        :param source: 待处理列表文件
        :param test_result_root: 保存预测所得标签的根目录
        """
        self.root = root
        self.source = source
        self.test_result_root = test_result_root
        self._data()

    def _data(self):
        """
        处理待测试数据的路径值
        :return:
        """
        self.neuron_name_list = []
        neuron_lines = open(self.source).readlines()
        neuron_lines = [line.strip() for line in neuron_lines]
        for line in neuron_lines:
            neuron_root = os.path.join(self.root, line)
            print(self.root)
            if not os.path.isdir(neuron_root):
                print(neuron_root)
                continue
            self.neuron_name_list.append(line)
        if len(self.neuron_name_list) == 0:
            raise AttributeError('图像路径中没有数据')

    def recover_size(self):
        """
        执行恢复大小 后处理
        :return:
        """
        start = datetime.now()
        for neuron_name in self.neuron_name_list:
            print('    recover the size of {}'.format(neuron_name))
            start_0 = datetime.now()
            neuron_name_root = os.path.join(self.root, neuron_name, 'image')
            neuron_data = Image3D_PATH(image_path = neuron_name_root)
            neuron_label_p_root = os.path.join(self.test_result_root, neuron_name, 'label_pre')
            pre_label = Image3D_PATH(image_path = neuron_label_p_root)
            pre_label.resize(shape_new = neuron_data.shape())
            pre_label.save(image_save_root = os.path.join(self.test_result_root, neuron_name, 'label_pre_same_size'), dim = 0)

            pre_label_1 = Image3D()
            pre_label_1.image_3d = deepcopy(pre_label.image_3d)
            pre_label_1.refresh_shape()
            pre_label_1.image_3d[pre_label_1.image_3d != 1] = 0
            pre_label_1.save(image_save_root = os.path.join(self.test_result_root, neuron_name, '1'), dim = 0)
            pre_label_2 = Image3D()
            pre_label_2.image_3d = deepcopy(pre_label.image_3d)
            pre_label_2.refresh_shape()
            pre_label_2.image_3d[pre_label_1.image_3d != 2] = 0
            pre_label_2.save(image_save_root = os.path.join(self.test_result_root, neuron_name, '2'), dim = 0)
            stop_0 = datetime.now()
            print('        took {} hours'.format(stop_0 - start_0))
        stop = datetime.now()
        print('took {} hours totally'.format(stop - start))

    def cut_from(self):
        """
        根据深度模型分割结果将复杂背景神经元图像对应目标神经元的部分切割出来
        :return:
        """
        start = datetime.now()
        for neuron_name in self.neuron_name_list:
            print('    cut {}'.format(neuron_name))
            start_0 = datetime.now()

            neuron_name_root = os.path.join(self.root, neuron_name, 'image')
            neuron_data = Image3D_PATH(image_path = neuron_name_root)
            neuron_label_p_1_root = os.path.join(self.test_result_root, neuron_name, '1')
            neuron_label_i_1_root = os.path.join(self.test_result_root, neuron_name, 'neuron_1')
            neuron_label_1 = Image3D_PATH(image_path = neuron_label_p_1_root)
            neuron_data.image_3d[neuron_label_1.image_3d != 1] = 0
            neuron_data.save(image_save_root = neuron_label_i_1_root, dim = 0)

            stop_0 = datetime.now()
            print(' ' *8 + 'took {} hours'.format(stop_0 - start_0))
        stop = datetime.now()
        print('took {} hours totally'.format(stop - start))

if __name__ == '__main__':
    """
    root = '/home/liqiufu/neuron_data/test_for_not_center'
    source = '/home/liqiufu/neuron_data/test_for_not_center/test_new.txt'
    root_saved = '/home/liqiufu/neuron_data/test_result'
    nt = Neuron_Test(root = root, source = source, root_saved = root_saved, batch = 1, depth = 32, height = 128, width = 128)
    nt.run()
    """

    #root = '/home/li-qiufu/PycharmProjects/MyDataBase/paper_data/0610'
    #source = '/home/li-qiufu/PycharmProjects/MyDataBase/paper_data/0610/result/test.txt'
    #test_result_root = '/home/li-qiufu/PycharmProjects/MyDataBase/paper_data/0610/result'
    #tpp = Test_Post_Processing(root = root, source = source, test_result_root = test_result_root)
    #tpp.recover_size()
    #tpp.cut_from()

    image_path = '/home/li-qiufu/PycharmProjects/MyDataBase/paper_data/0610/000000/image'
    label_path = '/home/li-qiufu/PycharmProjects/MyDataBase/paper_data/0610/000000/label'
    image_3d = Image3D_PATH(image_path = image_path)
    label_3d = Image3D_PATH(image_path = label_path)
    image_3d.image_3d[label_3d.image_3d != 1] = 0
    image_3d.save(image_save_root = '/home/li-qiufu/PycharmProjects/MyDataBase/paper_data/0610/000000/1')
