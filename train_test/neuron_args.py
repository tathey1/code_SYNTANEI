"""
这个脚本设置保存神经元图像训练过程中所用的参数值
"""
import os
import torch
from datetime import datetime
from tools.printer import Printer
from collections import OrderedDict
from networks.neuron_net import *

ARGS = OrderedDict()
ARGS['process_name'] = 'neuron_segmentation'        # 训练名称
#ARGS['comment'] = '使用 Model_2（3D UNet Plus）的预训练结果进行训练'
ARGS['comment'] = 'comment'
ARGS['net'] = UNet_3D

ARGS['root'] = '/cis/home/tathey/projects/mouselight/li_deep/DATA'        # 图像数据保存路径，与('train_root', 'test_root')相斥
ARGS['train_root'] = None                           # 训练数据保存路径，与'root'相斥
ARGS['test_root'] = None                            # 测试数据保存路径，与'root'相斥
ARGS['train_source'] = ARGS['root'] + '/train.txt'  # 训练图像列表文件
ARGS['test_source_0'] = ARGS['root'] + '/test.txt'    # 测试图像列表文件
ARGS['test_source_1'] = None   # 测试图像列表文件
ARGS['shuffle_train'] = True                        # 训练时，训练数据顺序是否打乱
ARGS['shuffle_test'] = False                        # 测试时，测试数据顺序是否打乱
ARGS['num_workers'] = 4                             # 读取数据时候，子进程的个数，当取值为0时，表示在主进程里读数据
ARGS['num_workers_test'] = 4                             # 读取数据时候，子进程的个数，当取值为0时，表示在主进程里读数据
ARGS['neuron_depth'] = 32                           # 训练和测试时神经元图像深度信息，即每个神经元 3D 图像由多少个分片合成
ARGS['neuron_height'] = 128                         # 训练和测试时候神经元图像的高度信息
ARGS['neuron_width'] = 128                          # 训练和测试时候神经元图像的宽度信息
                                                    # 神经元3D图像的尺寸格式为(depth, height, width)
                                                    # 尺寸不为(depth, height, width)的，需将其变换至这个数值
ARGS['gpu'] = [0]                               # 将使用的GPU显卡编号
ARGS['out_gpu'] = 0
ARGS['gpu_map'] = {'cuda:3':'cuda:0'}
                   #'cuda:6':'cuda:1'}
                   #'cuda:5':'cuda:2',
                   #'cuda:6':'cuda:3'}
ARGS['batch_size'] = 12                              # batch size 大小
ARGS['batch_size_test'] = 10                        # 测试时候的bat size大小
ARGS['epoch_number'] = 40                          # 训练周期数
ARGS['epoch_number_to_save_weight'] = 10             # 每隔多少个训练周期，保存一次训练权重，与'iter_number_to_save_weight'相斥
ARGS['iter_number_to_save_weight'] = None           # 每隔多少次batch迭代，保存一次训练权重，与'epoch_number_to_save_weight'相斥
ARGS['learning_rate'] = 0.007                       # 初始学习率
ARGS['poly_lr_decay_rate'] = 2                    # poly学习率衰减率
ARGS['lr_decay_rate'] = 0.75                         # step学习率衰减率
ARGS['lr_decay_epoch'] = 25                          # 每隔多少个迭代周期，衰减一次学习率，与'lr_decay_iter'相斥
ARGS['lr_decay_iter'] = None                        # 每隔多少个迭代周期，衰减一次学习率，与'lr_decay_epoch'相斥
ARGS['lr_decay_mode'] = 'poly'                      # step or poly
ARGS['iter_number_to_test'] = 200                   # 每隔多少次 iter，对测试数据进行一次测试
ARGS['loss_weight_cross_entropy'] = 1             # 损失函数中交叉熵权重
ARGS['loss_weight_mse'] = 0.0                       # 损失函数中欧氏距离权重

ARGS['weight_decay'] = 1e-4                         # 损失函数中正则项系数，权重衰减系数
ARGS['FL_gamma'] = 0.0000

ARGS['load_model'] = '/cis/home/tathey/projects/mouselight/li_deep/Model_info/Model_4/neuron_segmentation_U_63/neuron_segmentation_U_63_epoch_150.pkl'

                                                    # 已经预训练的权重文件
ARGS['class_weight'] = torch.Tensor([1.0, 33.0, 12]).float()     # 交叉熵损失函数中每一类损失的权重
# ARGS字典中设置为 None 的项暂时不可用

INFO_PATH = os.path.join('/cis/home/tathey/projects/mouselight/li_deep/NeuronImageProcess', 'info', ARGS['process_name'])
INFO_FILE = os.path.join(INFO_PATH, ARGS['process_name'] + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S.info'))

                                                # 日志文件名
WEIGHT_ROOT = os.path.join('/cis/home/tathey/projects/mouselight/li_deep/weights_model', ARGS['process_name'])
                                                # 网络权重数据保存路径


def _mkdirs():
    if not os.path.isdir(WEIGHT_ROOT):
        os.makedirs(WEIGHT_ROOT)
    if not os.path.isdir(INFO_PATH):
        os.makedirs(INFO_PATH)

def _print(printer):
    assert isinstance(printer, Printer)
    for key in ARGS:
        text = key + ': ' + str(ARGS[key])
        printer.pprint(text = text)
    text = 'INFO_FILE: ' + INFO_PATH
    printer.pprint(text = text)
    text = 'WEIGHT_ROOT: ' + WEIGHT_ROOT
    printer.pprint(text = text)
    printer.pprint(' ')
    printer.pprint(' ')
    printer.pprint(' ')
