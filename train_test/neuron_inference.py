import os
import torch
import numpy as np
from train_test.neuron_data import Neuron_Data_Set
from networks.neuron_net import Neuron_Net
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from datetime import datetime
from tools.image_fusion_in_spatial_domain import Image3D
from copy import deepcopy

def test(model_name = './neuron_pytorch_dl/weights_model/epoch_14_params.pkl'):
    save_root = './neuron_pytorch_dl/prediction_image'
    if not os.path.isdir(save_root):
        os.mkdir(save_root)
    torch.cuda.set_device(device = 7)
    net = Neuron_Net().cuda()
    net.load_state_dict(torch.load(model_name))
    net.eval()
    if not os.path.isdir('./prediction'):
        os.mkdir('./prediction')
    test_data_set = Neuron_Data_Set(root = '/data/liqiufu/neuron_data', source = '/data/liqiufu/neuron_data/test.txt', depth = 64, height = 256, width = 256)
    test_data_set_a = Neuron_Data_Set(root = '/data/liqiufu/neuron_data', source = '/data/liqiufu/neuron_data/test.txt', depth = 0, height = 0, width = 0)
    test_data_loader = DataLoader(dataset = test_data_set, batch_size = 1, shuffle = False, num_workers = 4)
    for index, ((image, label), (image_a, _)) in enumerate(zip(test_data_loader, test_data_set_a)):
        print('processing {}'.format(str(index)))
        start = datetime.now()
        pre_save_root = os.path.join(save_root, str(index).zfill(6))
        if not os.path.isdir(pre_save_root):
            os.mkdir(pre_save_root)
        else:
            os.system('rm ' + os.path.join(pre_save_root, '*'))
        pre_save_root_1 = os.path.join(pre_save_root, '1')
        pre_save_root_2 = os.path.join(pre_save_root, '2')
        if not os.path.isdir(pre_save_root_1):
            os.mkdir(pre_save_root_1)
        if not os.path.isdir(pre_save_root_2):
            os.mkdir(pre_save_root_2)
        image = Variable(image, requires_grad = False).cuda()
        output = net.forward(image)
        output = output.data.cpu().numpy().argmax(axis = 1)
        pre_3d = Image3D()
        pre_3d.image_3d = output.reshape(output.shape[1], output.shape[2], output.shape[3])
        pre_3d.refresh_shape()
        pre_3d.resize(shape_new = (image_a.shape[1], image_a.shape[2], image_a.shape[3]))
        print(pre_3d.shape())
        pre_3d_1 = deepcopy(pre_3d)
        pre_3d_1.image_3d[pre_3d_1.image_3d != 1] = 0
        pre_3d_2 = deepcopy(pre_3d)
        pre_3d_2.image_3d[pre_3d_2.image_3d != 2] = 0
        pre_3d_1.save(image_save_root = pre_save_root_1, dim = 0)
        pre_3d_2.save(image_save_root = pre_save_root_2, dim = 0)
        stop = datetime.now()
        print('    took {} hours'.format(stop - start))

if __name__ == '__main__':
    test()
