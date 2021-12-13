"""
重塑之前的预训练模型
"""
from networks.waveunet_3d import WaveUNet_3D_V2, WaveUNet_3D_V3, WaveUNet_3D_V4
import torch, os

class ReBuild():
    def __init__(self, pre_model, net, gpus = (0,)):
        self.net = net()
        if torch.cuda.is_available():
            self.net = self.net.cuda()
        self.pre_model = pre_model
        self._gpu(gpus)

    def _gpu(self, gpus):
        self.gpu = list(gpus)
        self.gpu_out = [0]
        self.gpu_map = dict()
        for i in range(len(gpus)):
            value = 'cuda:{}'.format(i)
            key = 'cuda:{}'.format(gpus[i])
            self.gpu_map[key] = value

    def save_model(self, save_root, filename):
        torch.save(self.net.state_dict(), os.path.join(save_root, filename))

    def rebuild(self):
        model_dict = self.net.state_dict()
        if torch.cuda.is_available():
            pretrained_dict = torch.load(self.pre_model, map_location = self.gpu_map)
        else:
            pretrained_dict = torch.load(self.pre_model, map_location = 'cpu')
        index = 0
        for key in model_dict:
            print('{} --- in self.net --- {}'.format(index, key))
            index += 1
        print('===================')
        index = 0
        for key in pretrained_dict:
            key_new = self.map_key_from_pre_model(key)
            flag_shape = model_dict[key_new].shape == pretrained_dict[key].shape if key_new in model_dict else False
            print('{} --- in pre_model --- {} / {} ---- {} ==> {}'.format(index,
                                                                          key_new in model_dict,
                                                                          flag_shape,
                                                                          self.map_key_from_pre_model(key).rjust(36),
                                                                          key))
            if key_new in model_dict and not flag_shape:
                print('{} != {}'.format(model_dict[key_new].shape, pretrained_dict[key].shape))
            index += 1

        pretrained_dict = [(self.map_key_from_pre_model(k), v) for k, v in pretrained_dict.items() if
                           self.map_key_from_pre_model(k) in model_dict and model_dict[self.map_key_from_pre_model(k)].shape == pretrained_dict[k].shape]
        print('====================')
        print('====================')
        for index, item in enumerate(pretrained_dict):
            print('{} ----- {}'.format(index, item[0]))
        model_dict.update(pretrained_dict)
        self.net.load_state_dict(model_dict)

    def map_key_from_pre_model(self, key):
        key_return = ''
        key_items = key.split('.')
        key_items_second = key_items[1].split('_')
        key0 = key_items_second[0]
        key1 = key_items_second[1]      # 11, 12, or 'r1'
        key2 = key_items[2]
        if key1 == '13':
            key_return = key_items[1] + '.' + key_items[2]
            return key_return
        if key0 == 't':
            key_return = key_items[1] + '.' + key_items[2]
            return key_return
        if key1.startswith('r') or key1 == 'input':
            assert key0 == 'con3d'
            key_return = 'con3d_' + key1 + '.' + key2
            return key_return
        if key0 == 'bn':
            key_return = '.'.join(['cdbr_' + key1, key0, key2])
            return key_return
        if key0 == 'cov3d' or key0 == 'con3d':
            key_return = '.'.join(['cdbr_' + key1, 'conv', key2])
            return key_return
        return key_return



if __name__ == '__main__':
    pre_model = '/home/liqiufu/PycharmProjects/NeuronImageProcess/neuron_pytorch_dl/weights_model/neuron_segmentation_U_70/neuron_segmentation_U_70_epoch_150.pkl'
    net = WaveUNet_3D_V4
    rebuild = ReBuild(pre_model = pre_model, net = net)
    rebuild.rebuild()
    save_root = '/home/liqiufu/PycharmProjects/NeuronImageProcess/pre_weights'
    filename = 'WaveUNet_3D_V4_from_U_70.pkl'
    rebuild.save_model(save_root = save_root, filename = filename)