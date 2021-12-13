"""
这个脚本对数据进行增强处理，将非背景坐标点位置的数据值修正为255
这个增强的想法是抹除目标神经元与干扰纤维由于数据值不同造成的差异，二者唯一差异在于目标神经元的全局特性和干扰纤维的局部性
"""
import os
from datetime import datetime
from train_test.neuron_data import Neuron_Data


def data_enhance(root, sub_pathes, angles, root_save):
    start_0 = datetime.now()
    for sub_path_ in sub_pathes:
        sub_path = os.path.join(root, sub_path_)
        sub_path_save = os.path.join(root_save, sub_path_)
        if not os.path.isdir(sub_path_save):
            os.mkdir(sub_path_save)
        for angle in angles:
            sub_path_angle = os.path.join(sub_path, angle)
            sub_path_angle_save = os.path.join(sub_path_save, angle)
            if not os.path.isdir(sub_path_angle_save):
                os.mkdir(sub_path_angle_save)

            neuron_name_list = os.listdir(sub_path_angle)
            neuron_name_list.sort()
            for neuron_name in neuron_name_list:
                start_1 = datetime.now()
                neuron_path = os.path.join(sub_path_angle, neuron_name)
                print('processing {}'.format(neuron_path))
                neuron_data = Neuron_Data(data_path = neuron_path)
                neuron_data.image.image_3d[neuron_data.label.image_3d != 0] = 255

                neuron_path_save = os.path.join(sub_path_angle_save, neuron_name)
                if not os.path.isdir(neuron_path_save):
                    os.mkdir(neuron_path_save)
                neuron_path_save_image = os.path.join(neuron_path_save, 'image')
                neuron_path_save_label = os.path.join(neuron_path_save, 'label')

                neuron_data.image.save(image_save_root = neuron_path_save_image, dim = 0)
                neuron_data.label.save(image_save_root = neuron_path_save_label, dim = 0)
                stop_1 = datetime.now()
                print('    took {} hours'.format(stop_1 - start_1))
                #key = input('input something')
    stop_0 = datetime.now()
    print('took {} hours totally'.format(stop_0 - start_0))


def data_resize(root, sub_pathes, angles, root_save):
    start_0 = datetime.now()
    for sub_path_ in sub_pathes:
        sub_path = os.path.join(root, sub_path_)
        sub_path_save = os.path.join(root_save, sub_path_)
        if not os.path.isdir(sub_path_save):
            os.mkdir(sub_path_save)
        for angle in angles:
            sub_path_angle = os.path.join(sub_path, angle)
            sub_path_angle_save = os.path.join(sub_path_save, angle)
            if not os.path.isdir(sub_path_angle_save):
                os.mkdir(sub_path_angle_save)

            neuron_name_list = os.listdir(sub_path_angle)
            neuron_name_list.sort()
            for neuron_name in neuron_name_list:
                start_1 = datetime.now()
                neuron_path = os.path.join(sub_path_angle, neuron_name)
                print('processing {}'.format(neuron_path))
                neuron_data = Neuron_Data(data_path = neuron_path)
                print(neuron_data.image.image_3d.shape)
                print(neuron_data.label.image_3d.shape)
                key = input('input something')
                neuron_data.resize(shape_new = (32,128,128))

                neuron_path_save = os.path.join(sub_path_angle_save, neuron_name)
                if not os.path.isdir(neuron_path_save):
                    os.mkdir(neuron_path_save)
                neuron_path_save_image = os.path.join(neuron_path_save, 'image')
                neuron_path_save_label = os.path.join(neuron_path_save, 'label')

                neuron_data.image.save(image_save_root = neuron_path_save_image, dim = 0)
                neuron_data.label.save(image_save_root = neuron_path_save_label, dim = 0)
                stop_1 = datetime.now()
                print('    took {} hours'.format(stop_1 - start_1))
    stop_0 = datetime.now()
    print('took {} hours totally'.format(stop_0 - start_0))


if __name__ == '__main__':
    root = '/home/li-qiufu/PycharmProjects/MyDataBase/neuron_data/neuron_data_2'
    sub_pathes = ['DataBase_16', 'DataBase_17', 'DataBase_18', 'DataBase_19', 'DataBase_20', 'DataBase_21']
    angles = ['angle_0', 'angle_60', 'angle_120', 'angle_180', 'angle_240', 'angle_300']

    root_save = '/home/li-qiufu/PycharmProjects/MyDataBase/neuron_data_32/neuron_data_2'
    data_resize(root, sub_pathes, angles, root_save)