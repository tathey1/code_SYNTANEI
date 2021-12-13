#!/usr/bin/env python
# -*- coding: utf-8 -*- 
"""
这个脚本描述使用神经元数据集合训练网络参数的过程，并提供接口
"""

import os
import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn.parallel import DataParallel
from torch.utils.data.dataloader import DataLoader
from datetime import datetime
from tools.printer import Printer
from train_test.neuron_label_compare import Compare_Segmentation_Label_CUDA_New as Compare_Segmentation_Label
from train_test.neuron_data import Neuron_Data_Set
from train_test.neuron_loss import CrossEntropy_MSE_Loss, CrossEntropy_Loss, FocalLoss, FocalLoss_
from copy import deepcopy
from train_test.neuron_args import ARGS, INFO_FILE, WEIGHT_ROOT, _print, _mkdirs


class Train_Test_Process():
    def __init__(self):
        _mkdirs()
        self.printer = Printer(INFO_FILE)
        _print(printer = self.printer)
        self.gpu_or_cpu = self._check_gpu()

        self.net = ARGS['net']()
        self.net.train()
        self.loss = CrossEntropy_Loss(w_crossentropy = 1, class_weight = ARGS['class_weight'].cuda())
        self.optimizer = self._optimizer()

        if self.gpu_or_cpu == 'gpu':
            self.net = DataParallel(module = self.net.cuda(), device_ids = ARGS['gpu'], output_device = ARGS['out_gpu'])
            self.loss = self.loss.cuda()
        if ARGS['load_model'] != None:
            self.load_pretrained_model()
        self._data()

    def _optimizer(self):
        weight_p, bias_p = [], []
        for name, p in self.net.named_parameters():
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        return optim.SGD([{'params': weight_p, 'weight_decay': ARGS['weight_decay']},
                          {'params': bias_p, 'weight_decay': 0}],
                         lr = ARGS['learning_rate'],
                         momentum = 0.9)

    def _data(self):
        """
        设置训练数据以及测试数据
        :return:
        """
        self.train_root = ARGS['root'] if ARGS['train_root'] == None else ARGS['train_root']
        self.test_root = ARGS['root'] if ARGS['test_root'] == None else ARGS['test_root']
        self.neuron_train_set = Neuron_Data_Set(root = self.train_root, source = ARGS['train_source'],
                                                depth = ARGS['neuron_depth'],
                                                height = ARGS['neuron_height'],
                                                width = ARGS['neuron_width'])
        print(self.neuron_train_set._neuron_name_list())
        self.train_data_loader = DataLoader(dataset = self.neuron_train_set, batch_size = ARGS['batch_size'],
                                            shuffle = ARGS['shuffle_train'], num_workers = ARGS['num_workers'])
        self.neuron_test_set_0 = Neuron_Data_Set(root = self.test_root, source = ARGS['test_source_0'],
                                               depth = ARGS['neuron_depth'],
                                               height = ARGS['neuron_height'],
                                               width = ARGS['neuron_width'])
        '''
        self.neuron_test_set_1 = Neuron_Data_Set(root = self.test_root, source = ARGS['test_source_1'],
                                               depth = ARGS['neuron_depth'],
                                               height = ARGS['neuron_height'],
                                               width = ARGS['neuron_width'])
        '''
        self.test_data_loader_0 = DataLoader(dataset = self.neuron_test_set_0, batch_size = ARGS['batch_size_test'],
                                           shuffle = ARGS['shuffle_test'], num_workers = ARGS['num_workers_test'])
        '''
        self.test_data_loader_1 = DataLoader(dataset = self.neuron_test_set_1, batch_size = ARGS['batch_size_test'],
                                           shuffle = ARGS['shuffle_test'], num_workers = ARGS['num_workers_test'])
        '''

    def _check_gpu(self):
        """
        检查确定是否可用GPU
        :return:
        """
        if torch.cuda.is_available():
            return 'gpu'
        else:
            return 'cpu'

    def load_pretrained_model(self):
        """
        加载预训练的模型参数，有可能某些模型参数被修改或增加减少某些层，这些情况应被处理
        :return:
        """
        model_dict = self.net.state_dict()
        pretrained_dict = torch.load(ARGS['load_model'], map_location = ARGS['gpu_map'])
        keys_abandon = [k for k in pretrained_dict if k not in model_dict]
        keys_without_load = [k for k in model_dict if k not in pretrained_dict]
        pretrained_dict = [(k, v) for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == pretrained_dict[k].shape]
        model_dict.update(pretrained_dict)
        self.net.load_state_dict(model_dict)
        for key in pretrained_dict:
            text = 'have loaded "{}" layer'.format(key[0])
            self.printer.pprint(text)
        for key in keys_abandon:
            text = '"{}" layer in pretrained model did not been loaded'.format(key)
            self.printer.pprint(text)
        for key in keys_without_load:
            text = '"{}" layer in current model did not been initilizated with pretrain model'.format(key)
            self.printer.pprint(text)

    def adjust_learning_rate(self, mode = ARGS['lr_decay_mode']):
        """
        调整学习率
        :return:
        """
        if mode == 'step':
            if (self.epoch + 1) % ARGS['lr_decay_epoch'] == 0:
                text = 'adjusting learning rate ...'
                self.printer.pprint(text)
            else:
                return
            decay_rate = ARGS['lr_decay_rate']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * decay_rate
        if mode == 'poly':
            decay_rate = (1 - self.epoch / (1.15 * ARGS['epoch_number'])) ** ARGS['poly_lr_decay_rate']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = ARGS['learning_rate'] * decay_rate

    def infer(self):
        """
        对测试数据进行测试
        :return:
        """
        #for test_loader in [self.test_data_loader_0, self.test_data_loader_1]:
        for test_loader in [self.test_data_loader_0,]:
            start = datetime.now()
            self.net.eval()
            length = len(test_loader)
            for index, (test_image, test_label) in enumerate(test_loader):
                start_1 = datetime.now()
                text = 'testing - {}/{}'.format(index, length)
                self.printer.pprint(text = ' ')
                self.printer.pprint(text = text)
                test_image = test_image.cuda()
                test_image.unsqueeze_(dim = 1)
                test_image = (test_image - 127.5) / 255
                print(test_image.shape)
                output0 = self.net.forward(test_image)
                print(output0.shape)
                torch.save(output0, "/cis/home/tathey/projects/mouselight/li_deep/DATA/output/output0.pt")
            stop = datetime.now()
            text = 'testing  took {} hours'.format(stop - start)
            self.printer.pprint(text = text)
        self.net.train()


    def test(self):
        """
        对测试数据进行测试
        :return:
        """
        #for test_loader in [self.test_data_loader_0, self.test_data_loader_1]:
        for test_loader in [self.test_data_loader_0,]:
            start = datetime.now()
            right_number = 0
            size_sum = 0
            loss_total = 0.0
            self.net.eval()
            length = len(test_loader)
            for index, (test_image, test_label) in enumerate(test_loader):
                start_1 = datetime.now()
                text = 'testing - {}/{}'.format(index, length)
                self.printer.pprint(text = ' ')
                self.printer.pprint(text = text)
                test_image = test_image.cuda()
                test_label = test_label.cuda()
                #print(test_label.size(), type(test_label))
                #print(torch.max(test_label))
                test_image.unsqueeze_(dim = 1)
                test_image = (test_image - 127.5) / 255
                output0 = self.net.forward(test_image)
                test_loss, str_loss = self.loss.forward(inputs = output0, targets = test_label)
                loss_total += test_loss.data
                _, predict_label = output0.topk(1, dim = 1)
                cls = Compare_Segmentation_Label(ground_truth = test_label,
                                                 predict_label = predict_label,
                                                 printer = self.printer,
                                                 model = 'test')
                cls.printing()
                stop_1 = datetime.now()
                text = 'test_loss = {:.6f} / {:.6f}, acc = {:.6f}, took {} hours'.format(test_loss, loss_total / (index+1), cls.accuracy, stop_1 - start_1)
                self.printer.pprint('testing - ' + text)
                self.printer.pprint(' ')

                right_number += cls.right_size
                size_sum += cls.size


                torch.save(output0, "/cis/home/tathey/projects/mouselight/li_deep/DATA/output/output_idx_" + str(index) + ".pt")
            self.printer.pprint('testing totally ---- ')
            cls.printing_()
            stop = datetime.now()
            text = 'testing - acc = {},  took {} hours'.format(right_number / size_sum, stop - start)
            self.printer.pprint(text = text)
        self.net.train()

    def train(self):
        """
        对数据进行训练
        :return:
        """
        start = datetime.now()
        number = 0
        train_length = len(self.train_data_loader)
        for epoch in range(ARGS['epoch_number']):
            start_0 = datetime.now()
            self.epoch = epoch
            train_loss_epoch = 0.0
            iteration_in_epoch = 0
            for index, (train_image, train_label) in enumerate(self.train_data_loader):
                start_1 = datetime.now()
                text = 'Iter {:6d}, Epoch {:3d}/{:3d}, batch {:4d} / {}, lr = {:.6f}'.format(number, epoch, ARGS['epoch_number'],
                                                                                             index, train_length,
                                                                                             self.optimizer.param_groups[0]['lr'])
                self.printer.pprint('training - ' + text)
                train_image.requires_grad_()
                train_image = train_image.cuda()
                train_label = train_label.cuda()
                train_image.unsqueeze_(dim = 1)
                train_image = (train_image - 127.5) / 255

                output0 = self.net.forward(train_image)
                train_loss, str_loss = self.loss.forward(inputs = output0, targets = train_label)
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                #self.printer.pprint(text = 'loss = {}'.format(str_loss))
                #self.printer.pprint(text = 'computing accuracy ...')
                _, predict_label = output0.topk(1, dim = 1)
                cls = Compare_Segmentation_Label(ground_truth = train_label,
                                                 predict_label = predict_label,
                                                 printer = self.printer)
                if number % 20 == 0:
                    cls.printing()
                train_loss_batch = train_loss.data
                acc = cls.accuracy

                train_loss_epoch += train_loss_batch
                stop_1 = datetime.now()
                text = 'train_loss = {:.6f} / {:.6f}, acc = {:.6f}, took {} hours -- {}'.format(train_loss_batch, train_loss_epoch / (iteration_in_epoch+1), acc, stop_1 - start_1, stop_1 - start)
                self.printer.pprint('training - ' + text)
                self.printer.pprint(' ')
                number += 1
                iteration_in_epoch += 1
                if (number - 8) % ARGS['iter_number_to_test'] == 0:
                    text = 'The {:2d}th test'.format(round((number + 1) / ARGS['iter_number_to_test']))
                    self.printer.pprint(' ')
                    self.printer.pprint('testing - ' + text)
                    self.test()
                    self.printer.pprint(' ')
            self.adjust_learning_rate()
            if (epoch + 1) % ARGS['epoch_number_to_save_weight'] == 0 or (epoch + 1) == ARGS['epoch_number']:
                text = 'saving weights ...'
                self.printer.pprint(text)
                torch.save(self.net.state_dict(),
                           os.path.join(WEIGHT_ROOT,
                                        ARGS['process_name'] + '_epoch_{}.pkl'.format(str(epoch + 1).zfill(3))))

            stop_0 = datetime.now()
            text = 'Epoch - {:3d}, train_loss = {:.6f}, took {} hours -- {}'.format(epoch, train_loss_epoch / iteration_in_epoch,
                                                                              stop_0 - start_0, stop_0 - start)
            self.printer.pprint(text)
            self.printer.pprint(' ')
        stop = datetime.now()
        text = 'train_test_process finish, took {} hours !!'.format(stop - start)
        self.printer.pprint(text)


if __name__ == '__main__':
    train_test_process = Train_Test_Process()
    train_test_process.train()

