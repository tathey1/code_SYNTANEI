# 这个脚本实现处理日志文件的一个功能，计算测试结果中各个项目的平均值

import numpy as np
import os
import matplotlib.pyplot as plt

def analyze(file_name, mode = 'train'):
    lines = (line.strip() for line in open(file_name).readlines())
    lines = (line for line in lines if len(line) > 26)
    lines = (line.split() for line in lines)
    mode = mode.lower()
    if mode == 'train_loss':
        loss = []
        for line in lines:
            if len(line) > 3 and line[3] == 'train_loss':
                loss.append(float(line[5][:-1]))
        plt.figure()
        plt.plot(range(len(loss)), loss, marker = '+', color = 'r', label = '3')
        plt.grid()
        plt.show()
    elif mode == 'test_loss':
        loss = []
        for line in lines:
            if len(line) > 3 and line[3] == 'test_loss':
                loss.append(float(line[5][:-1]))
        plt.figure()
        plt.plot(range(len(loss)), loss, marker = '+', color = 'r', label = '3')
        plt.grid()
        plt.show()
    else:
        rpi_0 = []
        rpi_1 = []
        rpi_2 = []
        lr = []
        for line in lines:
            if line[1] == '[' + mode + ']0_rpi':
                rpi_0.append(float(line[2]))
            if line[1] == '[' + mode + ']1_rpi':
                rpi_1.append(float(line[2]))
            if line[1] == '[' + mode + ']2_rpi':
                rpi_2.append(float(line[2]))
            if line[1] == 'training' and line[3] == 'Iter':
                lr.append(float(line[-1]))
        plt.figure()
        plt.subplot(221)
        plt.plot(range(len(rpi_0)), rpi_0, marker = '+', color = 'r', label = '3')
        plt.grid()
        plt.subplot(223)
        plt.plot(range(len(rpi_1)), rpi_1, marker = '+', color = 'c', label = '3')
        plt.grid()
        plt.subplot(222)
        plt.plot(range(len(rpi_2)), rpi_2, marker = '+', color = 'c', label = '3')
        plt.grid()
        plt.subplot(224)
        plt.plot(range(len(lr)), lr, marker = '+', color = 'c', label = '3')
        plt.grid()
        plt.show()

def process_info(file_name):
    """
    功能的具体实现
    :param file_name: 文件名
    :return:
    """
    line_list = open(file_name, 'r').readlines()
    line_list = [line.strip() for line in line_list]
    line_list = [line.split() for line in line_list]
    line_list = [line for line in line_list if len(line) >= 10]
    rpi = np.zeros((3,3))
    predicted_to = np.zeros((3,3))
    source_from = np.zeros((3,3))
    total_precision = np.zeros((1,2))
    No = 0
    for line in line_list:
        if line[1] == '[test]0_rpi':
            rpi[0,0] += float(line[ 2])
            rpi[0,1] += float(line[ 7])
            rpi[0,2] += float(line[12])
        if line[1] == '[test]1_rpi':
            rpi[1,0] += float(line[ 2])
            rpi[1,1] += float(line[ 7])
            rpi[1,2] += float(line[12])
        if line[1] == '[test]2_rpi':
            rpi[2,0] += float(line[ 2])
            rpi[2,1] += float(line[ 7])
            rpi[2,2] += float(line[12])

        if line[1] == '[test]0_predicted_to':
            predicted_to[0,0] += float(line[ 2])
            predicted_to[0,1] += float(line[ 7])
            predicted_to[0,2] += float(line[12])
        if line[1] == '[test]1_predicted_to':
            predicted_to[1,0] += float(line[ 2])
            predicted_to[1,1] += float(line[ 7])
            predicted_to[1,2] += float(line[12])
        if line[1] == '[test]2_predicted_to':
            predicted_to[2,0] += float(line[ 2])
            predicted_to[2,1] += float(line[ 7])
            predicted_to[2,2] += float(line[12])

        if line[1] == '[test]0_source_from':
            source_from[0, 0] += float(line[2])
            source_from[0, 1] += float(line[7])
            source_from[0, 2] += float(line[12])
        if line[1] == '[test]1_source_from':
            source_from[1,0] += float(line[ 2])
            source_from[1,1] += float(line[ 7])
            source_from[1,2] += float(line[12])
        if line[1] == '[test]2_source_from':
            source_from[2,0] += float(line[ 2])
            source_from[2,1] += float(line[ 7])
            source_from[2,2] += float(line[12])

        if line[1] == '[test]total_precision:':
            No += 1
            total_precision[0,0] += float(line[ 2])
            total_precision[0,1] += float(line[ 8])

    rpi = rpi / No
    predicted_to = predicted_to / No
    source_from = source_from / No
    total_precision = total_precision / No
    print(rpi)
    print('\n\n')
    print(predicted_to)
    print('\n\n')
    print(source_from)
    print('\n\n')
    print(total_precision)


class InfoProcessor():
    """
    这是一个训练对训练日志进行处理的类型
    """
    def __init__(self, info_name):
        self.info_name = info_name
        self.input = open(self.info_name)
        self.line = self.input.readline()
        self.flag = None            # 标志当前正在处理 train 数据还是 test 数据
        self.matrices_train = []
        self.matrices_test = []
        self.matrix_current_train = np.zeros((2,3,3,3))   # 存储某个train日志周期内的累计值
        self.matrix_current_test = np.zeros((2,3,3,3))    # 存储某个test日志周期内的累计值
        self.iter_train_current = 1         # 标志当前train周期内正在处理的迭代次数
        self.iter_test_current = 1          # 标志当前test周期内正在处理的迭代次数
        self.iteration_of_each_epoch_train()     # 一个train周期的迭代次数
        self.iteration_of_each_epoch_test()      # 一个test周期的迭代次数

    def iteration_of_each_epoch_train(self):
        """
        获取日志文件中每次训练周期的迭代次数
        :return:
        """
        while self.line:
            if 'training - Iter' in self.line:
                ele = self.line.split()
                self.iteration_train = int(ele[10][:-1])
                self.flag = 'train'
                self.updata_matrix_current()
                break
            self.line = self.input.readline()

    def iteration_of_each_epoch_test(self):

        """
        获取日志文件中每次测试周期的迭代次数
        这使得第一个训练周期内的数据没有被处理
        :return:
        """
        iteration_test = 0
        while self.line:
            if ' 1th test' in self.line:
                self.line = self.input.readline()
                self.line = self.input.readline()
                ele = self.line.split()
                self.iteration_test = int(ele[3][2:])
                self.flag = 'test'
                self.updata_matrix_current()
                break
            self.line = self.input.readline()

    def updata_matrix_current(self):
        """
        更新当前矩阵累计值，是train还是test，由 self.flag 标志
        :return:
        """
        #print('train for {}'.format(len(self.matrices_train)))
        #print('test for {}'.format(len(self.matrices_test)))
        #print('\n')
        if self.flag == None:
            raise AttributeError('标志位为空，需设置！')
        elif self.flag.lower() == 'train':
            while self.line:
                if self.iter_train_current == self.iteration_train:
                    self.update_matrices()
                    self.iter_train_current = 0
                    #self.flag = 'test'
                    break
                elif 'testing' in self.line and 'th test' in self.line:
                    self.flag = 'test'
                    break
                else:
                    ele = self.line.split()
                    if len(ele) <= 1:
                        self.line = self.input.readline()
                        continue
                    if ele[1] == '[train]0_rpi':
                        self.iter_train_current += 1
                        self.matrix_current_train[0,0,0,0] = int(ele[4])
                        self.matrix_current_train[1,0,0,0] = int(ele[6])
                        self.matrix_current_train[0,0,0,1] = int(ele[9])
                        self.matrix_current_train[1,0,0,1] = int(ele[11])
                        self.matrix_current_train[0,0,0,2] = int(ele[14])
                        self.matrix_current_train[1,0,0,2] = int(ele[16])
                    elif ele[1] == '[train]1_rpi':
                        self.matrix_current_train[0,0,1,0] = int(ele[4])
                        self.matrix_current_train[1,0,1,0] = int(ele[6])
                        self.matrix_current_train[0,0,1,1] = int(ele[9])
                        self.matrix_current_train[1,0,1,1] = int(ele[11])
                        self.matrix_current_train[0,0,1,2] = int(ele[14])
                        self.matrix_current_train[1,0,1,2] = int(ele[16])
                    elif ele[1] == '[train]2_rpi':
                        self.matrix_current_train[0,0,2,0] = int(ele[4])
                        self.matrix_current_train[1,0,2,0] = int(ele[6])
                        self.matrix_current_train[0,0,2,1] = int(ele[9])
                        self.matrix_current_train[1,0,2,1] = int(ele[11])
                        self.matrix_current_train[0,0,2,2] = int(ele[14])
                        self.matrix_current_train[1,0,2,2] = int(ele[16])
                    elif ele[1] == '[train]0_predicted_to':
                        self.matrix_current_train[0,1,0,0] = int(ele[4])
                        self.matrix_current_train[1,1,0,0] = int(ele[6])
                        self.matrix_current_train[0,1,0,1] = int(ele[9])
                        self.matrix_current_train[1,1,0,1] = int(ele[11])
                        self.matrix_current_train[0,1,0,2] = int(ele[14])
                        self.matrix_current_train[1,1,0,2] = int(ele[16])
                    elif ele[1] == '[train]1_predicted_to':
                        self.matrix_current_train[0,1,1,0] = int(ele[4])
                        self.matrix_current_train[1,1,1,0] = int(ele[6])
                        self.matrix_current_train[0,1,1,1] = int(ele[9])
                        self.matrix_current_train[1,1,1,1] = int(ele[11])
                        self.matrix_current_train[0,1,1,2] = int(ele[14])
                        self.matrix_current_train[1,1,1,2] = int(ele[16])
                    elif ele[1] == '[train]2_predicted_to':
                        self.matrix_current_train[0,1,2,0] = int(ele[4])
                        self.matrix_current_train[1,1,2,0] = int(ele[6])
                        self.matrix_current_train[0,1,2,1] = int(ele[9])
                        self.matrix_current_train[1,1,2,1] = int(ele[11])
                        self.matrix_current_train[0,1,2,2] = int(ele[14])
                        self.matrix_current_train[1,1,2,2] = int(ele[16])
                    elif ele[1] == '[train]0_source_from':
                        self.matrix_current_train[0,2,0,0] = int(ele[4])
                        self.matrix_current_train[1,2,0,0] = int(ele[6])
                        self.matrix_current_train[0,2,0,1] = int(ele[9])
                        self.matrix_current_train[1,2,0,1] = int(ele[11])
                        self.matrix_current_train[0,2,0,2] = int(ele[14])
                        self.matrix_current_train[1,2,0,2] = int(ele[16])
                    elif ele[1] == '[train]1_source_from':
                        self.matrix_current_train[0,2,1,0] = int(ele[4])
                        self.matrix_current_train[1,2,1,0] = int(ele[6])
                        self.matrix_current_train[0,2,1,1] = int(ele[9])
                        self.matrix_current_train[1,2,1,1] = int(ele[11])
                        self.matrix_current_train[0,2,1,2] = int(ele[14])
                        self.matrix_current_train[1,2,1,2] = int(ele[16])
                    elif ele[1] == '[train]2_source_from':
                        self.matrix_current_train[0,2,2,0] = int(ele[4])
                        self.matrix_current_train[1,2,2,0] = int(ele[6])
                        self.matrix_current_train[0,2,2,1] = int(ele[9])
                        self.matrix_current_train[1,2,2,1] = int(ele[11])
                        self.matrix_current_train[0,2,2,2] = int(ele[14])
                        self.matrix_current_train[1,2,2,2] = int(ele[16])
                self.line = self.input.readline()
        elif self.flag.lower() == 'test':
            while self.line:
                if self.iter_test_current == self.iteration_test:
                    print(self.iter_test_current, self.iteration_test, len(self.matrices_test))
                    self.update_matrices()
                    self.iter_test_current = 0
                    self.flag = 'train'
                    break
                else:
                    ele = self.line.split()
                    if len(ele) <= 1:
                        self.line = self.input.readline()
                        continue
                    if ele[1] == '[test]0_rpi':
                        self.iter_test_current += 1
                        self.matrix_current_test[0,0,0,0] = int(ele[4])
                        self.matrix_current_test[1,0,0,0] = int(ele[6])
                        self.matrix_current_test[0,0,0,1] = int(ele[9])
                        self.matrix_current_test[1,0,0,1] = int(ele[11])
                        self.matrix_current_test[0,0,0,2] = int(ele[14])
                        self.matrix_current_test[1,0,0,2] = int(ele[16])
                    elif ele[1] == '[test]1_rpi':
                        self.matrix_current_test[0,0,1,0] = int(ele[4])
                        self.matrix_current_test[1,0,1,0] = int(ele[6])
                        self.matrix_current_test[0,0,1,1] = int(ele[9])
                        self.matrix_current_test[1,0,1,1] = int(ele[11])
                        self.matrix_current_test[0,0,1,2] = int(ele[14])
                        self.matrix_current_test[1,0,1,2] = int(ele[16])
                    elif ele[1] == '[test]2_rpi':
                        self.matrix_current_test[0,0,2,0] = int(ele[4])
                        self.matrix_current_test[1,0,2,0] = int(ele[6])
                        self.matrix_current_test[0,0,2,1] = int(ele[9])
                        self.matrix_current_test[1,0,2,1] = int(ele[11])
                        self.matrix_current_test[0,0,2,2] = int(ele[14])
                        self.matrix_current_test[1,0,2,2] = int(ele[16])
                    elif ele[1] == '[test]0_predicted_to':
                        self.matrix_current_test[0,1,0,0] = int(ele[4])
                        self.matrix_current_test[1,1,0,0] = int(ele[6])
                        self.matrix_current_test[0,1,0,1] = int(ele[9])
                        self.matrix_current_test[1,1,0,1] = int(ele[11])
                        self.matrix_current_test[0,1,0,2] = int(ele[14])
                        self.matrix_current_test[1,1,0,2] = int(ele[16])
                    elif ele[1] == '[test]1_predicted_to':
                        self.matrix_current_test[0,1,1,0] = int(ele[4])
                        self.matrix_current_test[1,1,1,0] = int(ele[6])
                        self.matrix_current_test[0,1,1,1] = int(ele[9])
                        self.matrix_current_test[1,1,1,1] = int(ele[11])
                        self.matrix_current_test[0,1,1,2] = int(ele[14])
                        self.matrix_current_test[1,1,1,2] = int(ele[16])
                    elif ele[1] == '[test]2_predicted_to':
                        self.matrix_current_test[0,1,2,0] = int(ele[4])
                        self.matrix_current_test[1,1,2,0] = int(ele[6])
                        self.matrix_current_test[0,1,2,1] = int(ele[9])
                        self.matrix_current_test[1,1,2,1] = int(ele[11])
                        self.matrix_current_test[0,1,2,2] = int(ele[14])
                        self.matrix_current_test[1,1,2,2] = int(ele[16])
                    elif ele[1] == '[test]0_source_from':
                        self.matrix_current_test[0,2,0,0] = int(ele[4])
                        self.matrix_current_test[1,2,0,0] = int(ele[6])
                        self.matrix_current_test[0,2,0,1] = int(ele[9])
                        self.matrix_current_test[1,2,0,1] = int(ele[11])
                        self.matrix_current_test[0,2,0,2] = int(ele[14])
                        self.matrix_current_test[1,2,0,2] = int(ele[16])
                    elif ele[1] == '[test]1_source_from':
                        self.matrix_current_test[0,2,1,0] = int(ele[4])
                        self.matrix_current_test[1,2,1,0] = int(ele[6])
                        self.matrix_current_test[0,2,1,1] = int(ele[9])
                        self.matrix_current_test[1,2,1,1] = int(ele[11])
                        self.matrix_current_test[0,2,1,2] = int(ele[14])
                        self.matrix_current_test[1,2,1,2] = int(ele[16])
                    elif ele[1] == '[test]2_source_from':
                        self.matrix_current_test[0,2,2,0] = int(ele[4])
                        self.matrix_current_test[1,2,2,0] = int(ele[6])
                        self.matrix_current_test[0,2,2,1] = int(ele[9])
                        self.matrix_current_test[1,2,2,1] = int(ele[11])
                        self.matrix_current_test[0,2,2,2] = int(ele[14])
                        self.matrix_current_test[1,2,2,2] = int(ele[16])
                self.line = self.input.readline()

    def update_matrices(self):
        """
        将已经处理了一个周期的数据加工后保存
        :return:
        """
        temp = np.zeros((3,3,3))
        if self.flag.lower() == 'train':
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        temp[i,j,k] = self.matrix_current_train[0,i,j,k] / self.matrix_current_train[1,i,j,k]
            self.matrices_train.append(temp)
            self.matrix_current_train = np.zeros((2,3,3,3))
        if self.flag.lower() == 'test':
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        temp[i,j,k] = self.matrix_current_test[0,i,j,k] / self.matrix_current_test[1,i,j,k]
            self.matrices_test.append(temp)
            self.matrix_current_test = np.zeros((2,3,3,3))
        return

    def run(self):
        """
        处理数据
        :return:
        """
        while self.line:
            self.updata_matrix_current()
        self.save()
        return

    def save(self):
        """
        保存处理所得数据
        :return:
        """
        train_file_name = os.path.splitext(self.info_name)[0] + '_train.txt'
        test_file_name = os.path.splitext(self.info_name)[0] + '_test.txt'
        train_file = open(train_file_name, 'w')
        test_file = open(test_file_name, 'w')
        matrices_train = list(zip(range(len(self.matrices_train)), self.matrices_train))
        matrices_train.sort(key = lambda x:(x[1][0,1,0]-x[1][1,2,1]))
        for index, matrix in matrices_train:
            text = '=' * 10 + str(index+1).zfill(4) + '=' * 10 + '\n'
            train_file.write(text)
            for i in range(3):
                text = '-' * 20 + '\n'
                train_file.write(text)
                for j in range(3):
                    text = ''
                    for k in range(3):
                        text += str(round(matrix[i,j,k],4)) + '\t'
                    text += '\n'
                    train_file.write(text)
            text = '=' * 20 + '\n'
            train_file.write(text + '\n\n')
        matrices_test = list(zip(range(len(self.matrices_test)), self.matrices_test))
        matrices_test.sort(key = lambda x:(x[1][0,1,0]-x[1][1,2,1]))
        for index, matrix in matrices_test:
            text = '=' * 10 + str(index+1).zfill(4) + '=' * 10 + '\n'
            test_file.write(text)
            for i in range(3):
                text = '-' * 20 + '\n'
                test_file.write(text)
                for j in range(3):
                    text = ''
                    for k in range(3):
                        text += str(round(matrix[i,j,k],4)) + '\t'
                    text += '\n'
                    test_file.write(text)
            text = '=' * 20 + '\n'
            test_file.write(text + '\n\n')
        train_file.close()
        test_file.close()


def main():
    root = '/home/li-qiufu/PycharmProjects/NeuronImageProcess/neuron_pytorch_dl/info'
    file_list = os.listdir(root)
    file_list.sort()
    for file_name in file_list:
        ele = file_name.split('_')
        if ele[-1].isdigit() and int(ele[-1]) >= 70:
            subpath = os.path.join(root, file_name)
            sub_file_list = os.listdir(subpath)
            for sub_file_name in sub_file_list:
                sub_file_name_full = os.path.join(subpath, sub_file_name)
                if not sub_file_name_full[-4:] == 'info':
                    continue
                print(sub_file_name_full)
                IP = InfoProcessor(info_name = sub_file_name_full)
                IP.run()

if __name__ == '__main__':
    info_name = '/home/li-qiufu/PycharmProjects/NeuronImageProcess/neuron_pytorch_dl/info/test_result.txt'
    IP = InfoProcessor(info_name = info_name)
    #analyze(info_name, mode = 'test')
    IP.run()