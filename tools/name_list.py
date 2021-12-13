import os
"""
root = '/home/liqiufu/neuron_data/neuron_data_augmentation'
actions = ['neuron_data_1', 'neuron_data_1_x', 'neuron_data_1_y', 'neuron_data_noise', 'neuron_data_noise_x', 'neuron_data_noise_y']
data_bases = ['DataBase_16', 'DataBase_17', 'DataBase_18', 'DataBase_19', 'DataBase_20', 'DataBase_21']
"""
root = '/home/li-qiufu/PycharmProjects/MyDataBase/1_HUST_DATA'
actions = ['HUST_DATA', 'HUST_DATA_noise', 'HUST_DATA_noise_x', 'HUST_DATA_noise_y', 'HUST_DATA_x', 'HUST_DATA_y']
data_bases = ['']
train_txt = os.path.join(root, 'train.txt')
test_txt = os.path.join(root, 'test.txt')

def name_list():
    train_info = open(train_txt, 'w')
    test_info = open(test_txt, 'w')
    for action in actions:
        for data_base in data_bases:
            path = os.path.join(root, action, data_base)
            sub_path_list = os.listdir(path)
            sub_path_list.sort()
            for sub_path_name in sub_path_list:
                sub_path = os.path.join(path, sub_path_name)
                if not os.path.isdir(sub_path):
                    continue
                object_list = os.listdir(sub_path)
                object_list.sort()
                for index, object_ in enumerate(object_list):
                    if not os.path.isdir(os.path.join(sub_path, object_)):
                        continue
                    #if (index + 1) % 4 == 0:
                    test_info.write(os.path.join(action, data_base, sub_path_name, object_) + '\n')
                    #else:
                    #    train_info.write(os.path.join(action, data_base, sub_path_name, object_) + '\n')
    train_info.close()
    test_info.close()

if __name__ == '__main__':
    name_list()
