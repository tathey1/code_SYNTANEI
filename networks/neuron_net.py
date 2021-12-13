"""
这个脚本描述用于训练神经元图像的网络结构
这个几个网络模型可以写成一个通用类型，但我比较懒，不想改了
"""

import torch
from torch import nn
from torch.nn import Module


class UNet_3D(Module):
    """
    Model 0
    """
    def __init__(self):
        super(UNet_3D, self).__init__()
        # 32 * 128 * 128
        self.cov3d_11 = nn.Conv3d(in_channels = 1, out_channels = 16, kernel_size = 3, padding = 1)
        self.bn_11 = nn.BatchNorm3d(num_features = 16)
        self.relu_11 = nn.ReLU(inplace = False)
        self.cov3d_12 = nn.Conv3d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = 1)
        self.bn_12 = nn.BatchNorm3d(num_features = 16)
        self.relu_12 = nn.ReLU(inplace = False)
        self.maxpool_1 = nn.MaxPool3d(kernel_size = 2)

        # 16 * 64 * 64
        self.cov3d_21 = nn.Conv3d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1)
        self.bn_21 = nn.BatchNorm3d(num_features = 32)
        self.relu_21 = nn.ReLU(inplace = False)
        self.cov3d_22 = nn.Conv3d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1)
        self.bn_22 = nn.BatchNorm3d(num_features = 32)
        self.relu_22 = nn.ReLU(inplace = False)
        self.maxpool_2 = nn.MaxPool3d(kernel_size = 2)

        # 8 * 32 * 32
        self.cov3d_31 = nn.Conv3d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
        self.bn_31 = nn.BatchNorm3d(num_features = 64)
        self.relu_31 = nn.ReLU(inplace = False)
        self.cov3d_32 = nn.Conv3d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
        self.bn_32 = nn.BatchNorm3d(num_features = 64)
        self.relu_32 = nn.ReLU(inplace = False)
        self.maxpool_3 = nn.MaxPool3d(kernel_size = 2)

        #  4 * 16 * 16
        self.cov3d_41 = nn.Conv3d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
        self.bn_41 = nn.BatchNorm3d(num_features = 128)
        self.relu_41 = nn.ReLU(inplace = False)
        self.cov3d_42 = nn.Conv3d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
        self.bn_42 = nn.BatchNorm3d(num_features = 128)
        self.relu_42 = nn.ReLU(inplace = False)

        self.maxpool_4 = nn.MaxPool3d(kernel_size = 2)

        self.con3d_51 = nn.Conv3d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1)
        self.bn_51 = nn.BatchNorm3d(num_features = 256)
        self.relu_51 = nn.ReLU(inplace = False)
        self.con3d_52 = nn.Conv3d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        self.drop_52 = nn.Dropout3d(p = 0.5)
        self.bn_52 = nn.BatchNorm3d(num_features = 256)
        self.relu_52 = nn.ReLU(inplace = False)

        # 4 * 16 * 16
        self.t_cov3d_9 =nn.ConvTranspose3d(in_channels = 256, out_channels = 128, kernel_size = 3, stride = 2, output_padding = 1, padding = 1)
        self.cov3d_91_ = nn.Conv3d(in_channels = 256, out_channels = 128, kernel_size = 3, padding = 1)
        self.bn_91 = nn.BatchNorm3d(num_features = 128)
        self.relu_91 = nn.ReLU(inplace = False)
        self.cov3d_92 = nn.Conv3d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
        self.bn_92 = nn.BatchNorm3d(num_features = 128)
        self.relu_92 = nn.ReLU(inplace = False)

        # 8 * 32 * 32
        self.t_cov3d_10 = nn.ConvTranspose3d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 2, output_padding = 1, padding = 1)
        self.cov3d_101 = nn.Conv3d(in_channels = 128, out_channels = 64, kernel_size = 3, padding = 1)
        self.bn_101 = nn.BatchNorm3d(num_features = 64)
        self.relu_101 = nn.ReLU(inplace = False)
        self.cov3d_102 = nn.Conv3d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
        self.bn_102 = nn.BatchNorm3d(num_features = 64)
        self.relu_102 = nn.ReLU(inplace = False)

        # 16 * 64 * 64
        self.t_cov3d_11 = nn.ConvTranspose3d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 2, output_padding = 1, padding = 1)
        self.cov3d_111 = nn.Conv3d(in_channels = 64, out_channels = 32, kernel_size = 3, padding = 1)
        self.bn_111 = nn.BatchNorm3d(num_features = 32)
        self.relu_111 = nn.ReLU(inplace = False)
        self.cov3d_112 = nn.Conv3d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1)
        self.bn_112 = nn.BatchNorm3d(num_features = 32)
        self.relu_112 = nn.ReLU(inplace = False)

        # 32 * 128 * 128
        self.t_cov3d_12 = nn.ConvTranspose3d(in_channels = 32, out_channels = 16, kernel_size = 3, stride = 2, output_padding = 1, padding = 1)
        self.cov3d_121 = nn.Conv3d(in_channels = 32, out_channels = 16, kernel_size = 3, padding = 1)
        self.bn_121 = nn.BatchNorm3d(num_features = 16)
        self.relu_121 = nn.ReLU(inplace = False)
        self.cov3d_122 = nn.Conv3d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = 1)
        self.bn_122 = nn.BatchNorm3d(num_features = 16)
        self.relu_122 = nn.ReLU(inplace = False)

        # 128 * 128
        self.cov3d_13_ = nn.Conv3d(in_channels = 16, out_channels = 3, kernel_size = 1)

    def forward(self, input):
        print(f"input shape: {input.shape}")
        # 32 * 128 * 128
        output = self.cov3d_11(input)
        output = self.bn_11(output)
        output = self.relu_11(output)
        output = self.cov3d_12(output)
        output = self.bn_12(output)
        output1 = self.relu_12(output)
        output = self.maxpool_1(output1)
        print(f"outputshape: {output.shape}")

        # 16 * 64 * 64
        output = self.cov3d_21(output)
        output = self.bn_21(output)
        output = self.relu_21(output)
        output = self.cov3d_22(output)
        output = self.bn_22(output)
        output2 = self.relu_22(output)
        output = self.maxpool_2(output2)
        print(f"outputshape: {output.shape}")
        
        # 8 * 32 * 32
        output = self.cov3d_31(output)
        output = self.bn_31(output)
        output = self.relu_31(output)
        output = self.cov3d_32(output)
        output = self.bn_32(output)
        output3 = self.relu_32(output)
        output = self.maxpool_3(output3)
        print(f"outputshape: {output.shape}")

        # 4 * 16 * 16
        output = self.cov3d_41(output)
        output = self.bn_41(output)
        output = self.relu_41(output)
        output = self.cov3d_42(output)
        output = self.bn_42(output)
        output4 = self.relu_42(output)
        print(f"output shape: {output.shape} to output4 shape: {output4.shape}")
        output = self.maxpool_4(output4)

        output = self.con3d_51(output)
        output = self.bn_51(output)
        output = self.relu_51(output)
        output = self.con3d_52(output)
        output = self.drop_52(output)
        output = self.bn_52(output)
        output5 = self.relu_52(output)
        
        # 4 * 16 * 16
        output = self.t_cov3d_9(output5)
        print(f"output5 shape {output5.shape} to output shape: {output.shape}")
        print(f"outputshape: {output.shape}")
        print(f"output4shape: {output4.shape}")
        output = torch.cat((output, output4), dim = 1)
        output = self.cov3d_91_(output)
        output = self.bn_91(output)
        output = self.relu_91(output)
        output = self.cov3d_92(output)
        output = self.bn_92(output)
        output9 = self.relu_92(output)

        # 8 * 32 * 32
        output = self.t_cov3d_10(output9)
        output = torch.cat((output, output3), dim = 1)
        output = self.cov3d_101(output)
        output = self.bn_101(output)
        output = self.relu_101(output)
        output = self.cov3d_102(output)
        output = self.bn_102(output)
        output10 = self.relu_102(output)

        # 16 * 64 * 64
        output = self.t_cov3d_11(output10)
        output = torch.cat((output, output2), dim = 1)
        output = self.cov3d_111(output)
        output = self.bn_111(output)
        output = self.relu_111(output)
        output = self.cov3d_112(output)
        output = self.bn_112(output)
        output11 = self.relu_112(output)

        # 32 * 128 * 128
        output = self.t_cov3d_12(output11)
        output = torch.cat((output, output1), dim = 1)
        output = self.cov3d_121(output)
        output = self.bn_121(output)
        output = self.relu_121(output)
        output = self.cov3d_122(output)
        output = self.bn_122(output)
        output12 = self.relu_122(output)

        output = self.cov3d_13_(output12)

        return output


class UNet_3D_D(Module):
    """
    Model 3
    """
    def __init__(self):
        super(UNet_3D_D, self).__init__()
        # 32 * 128 * 128
        self.cov3d_11 = nn.Conv3d(in_channels = 1, out_channels = 16, kernel_size = 3, padding = 1)
        self.drop_11 = nn.Dropout3d(p = 0.1)
        self.bn_11 = nn.BatchNorm3d(num_features = 16)
        self.relu_11 = nn.ReLU(inplace = False)
        self.cov3d_12 = nn.Conv3d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = 1)
        self.drop_12 = nn.Dropout3d(p = 0.1)
        self.bn_12 = nn.BatchNorm3d(num_features = 16)
        self.relu_12 = nn.ReLU(inplace = False)
        self.maxpool_1 = nn.MaxPool3d(kernel_size = 2)

        # 16 * 64 * 64
        self.cov3d_21 = nn.Conv3d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1)
        self.drop_21 = nn.Dropout3d(p = 0.1)
        self.bn_21 = nn.BatchNorm3d(num_features = 32)
        self.relu_21 = nn.ReLU(inplace = False)
        self.cov3d_22 = nn.Conv3d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1)
        self.drop_22 = nn.Dropout3d(p = 0.1)
        self.bn_22 = nn.BatchNorm3d(num_features = 32)
        self.relu_22 = nn.ReLU(inplace = False)
        self.maxpool_2 = nn.MaxPool3d(kernel_size = 2)

        # 8 * 32 * 32
        self.cov3d_31 = nn.Conv3d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
        self.drop_31 = nn.Dropout3d(p = 0.15)
        self.bn_31 = nn.BatchNorm3d(num_features = 64)
        self.relu_31 = nn.ReLU(inplace = False)
        self.cov3d_32 = nn.Conv3d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
        self.drop_32 = nn.Dropout3d(p = 0.15)
        self.bn_32 = nn.BatchNorm3d(num_features = 64)
        self.relu_32 = nn.ReLU(inplace = False)
        self.maxpool_3 = nn.MaxPool3d(kernel_size = 2)

        #  4 * 16 * 16
        self.cov3d_41 = nn.Conv3d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
        self.drop_41 = nn.Dropout3d(p = 0.2)
        self.bn_41 = nn.BatchNorm3d(num_features = 128)
        self.relu_41 = nn.ReLU(inplace = False)
        self.cov3d_42 = nn.Conv3d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
        self.drop_42 = nn.Dropout3d(p = 0.2)
        self.bn_42 = nn.BatchNorm3d(num_features = 128)
        self.relu_42 = nn.ReLU(inplace = False)

        self.maxpool_4 = nn.MaxPool3d(kernel_size = 2)

        self.con3d_51 = nn.Conv3d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1)
        self.drop_51 = nn.Dropout3d(p = 0.25)
        self.bn_51 = nn.BatchNorm3d(num_features = 256)
        self.relu_51 = nn.ReLU(inplace = False)
        self.con3d_52 = nn.Conv3d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        self.drop_52 = nn.Dropout3d(p = 0.25)
        self.bn_52 = nn.BatchNorm3d(num_features = 256)
        self.relu_52 = nn.ReLU(inplace = False)

        # 4 * 16 * 16
        self.t_cov3d_9 =nn.ConvTranspose3d(in_channels = 256, out_channels = 128, kernel_size = 3, stride = 2, output_padding = 1, padding = 1)
        self.cov3d_91_ = nn.Conv3d(in_channels = 256, out_channels = 128, kernel_size = 3, padding = 1)
        self.drop_91 = nn.Dropout3d(p = 0.25)
        self.bn_91 = nn.BatchNorm3d(num_features = 128)
        self.relu_91 = nn.ReLU(inplace = False)
        self.cov3d_92 = nn.Conv3d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
        self.drop_92 = nn.Dropout3d(p = 0.25)
        self.bn_92 = nn.BatchNorm3d(num_features = 128)
        self.relu_92 = nn.ReLU(inplace = False)

        # 8 * 32 * 32
        self.t_cov3d_10 = nn.ConvTranspose3d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 2, output_padding = 1, padding = 1)
        self.cov3d_101 = nn.Conv3d(in_channels = 128, out_channels = 64, kernel_size = 3, padding = 1)
        self.drop_101 = nn.Dropout3d(p = 0.2)
        self.bn_101 = nn.BatchNorm3d(num_features = 64)
        self.relu_101 = nn.ReLU(inplace = False)
        self.cov3d_102 = nn.Conv3d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
        self.drop_102 = nn.Dropout3d(p = 0.2)
        self.bn_102 = nn.BatchNorm3d(num_features = 64)
        self.relu_102 = nn.ReLU(inplace = False)

        # 16 * 64 * 64
        self.t_cov3d_11 = nn.ConvTranspose3d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 2, output_padding = 1, padding = 1)
        self.cov3d_111 = nn.Conv3d(in_channels = 64, out_channels = 32, kernel_size = 3, padding = 1)
        self.drop_111 = nn.Dropout3d(p = 0.1)
        self.bn_111 = nn.BatchNorm3d(num_features = 32)
        self.relu_111 = nn.ReLU(inplace = False)
        self.cov3d_112 = nn.Conv3d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1)
        self.drop_112 = nn.Dropout3d(p = 0.1)
        self.bn_112 = nn.BatchNorm3d(num_features = 32)
        self.relu_112 = nn.ReLU(inplace = False)

        # 32 * 128 * 128
        self.t_cov3d_12 = nn.ConvTranspose3d(in_channels = 32, out_channels = 16, kernel_size = 3, stride = 2, output_padding = 1, padding = 1)
        self.cov3d_121 = nn.Conv3d(in_channels = 32, out_channels = 16, kernel_size = 3, padding = 1)
        self.drop_121 = nn.Dropout3d(p = 0.1)
        self.bn_121 = nn.BatchNorm3d(num_features = 16)
        self.relu_121 = nn.ReLU(inplace = False)
        self.cov3d_122 = nn.Conv3d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = 1)
        self.drop_122 = nn.Dropout3d(p = 0.1)
        self.bn_122 = nn.BatchNorm3d(num_features = 16)
        self.relu_122 = nn.ReLU(inplace = False)

        # 128 * 128
        self.cov3d_13 = nn.Conv3d(in_channels = 16, out_channels = 3, kernel_size = 1)

    def forward(self, input):
        # 32 * 128 * 128
        output = self.cov3d_11(input)
        output = self.drop_11(output)
        output = self.bn_11(output)
        output = self.relu_11(output)
        output = self.cov3d_12(output)
        output = self.drop_12(output)
        output = self.bn_12(output)
        output1 = self.relu_12(output)
        output = self.maxpool_1(output1)

        # 16 * 64 * 64
        output = self.cov3d_21(output)
        output = self.drop_21(output)
        output = self.bn_21(output)
        output = self.relu_21(output)
        output = self.cov3d_22(output)
        output = self.drop_22(output)
        output = self.bn_22(output)
        output2 = self.relu_22(output)
        output = self.maxpool_2(output2)

        # 8 * 32 * 32
        output = self.cov3d_31(output)
        output = self.drop_31(output)
        output = self.bn_31(output)
        output = self.relu_31(output)
        output = self.cov3d_32(output)
        output = self.drop_32(output)
        output = self.bn_32(output)
        output3 = self.relu_32(output)
        output = self.maxpool_3(output3)

        # 4 * 16 * 16
        output = self.cov3d_41(output)
        output = self.drop_41(output)
        output = self.bn_41(output)
        output = self.relu_41(output)
        output = self.cov3d_42(output)
        output = self.drop_42(output)
        output = self.bn_42(output)
        output4 = self.relu_42(output)
        output = self.maxpool_4(output4)

        output = self.con3d_51(output)
        output = self.drop_51(output)
        output = self.bn_51(output)
        output = self.relu_51(output)
        output = self.con3d_52(output)
        output = self.drop_52(output)
        output = self.bn_52(output)
        output5 = self.relu_52(output)

        # 4 * 16 * 16
        output = self.t_cov3d_9(output5)
        output = torch.cat((output, output4), dim = 1)
        output = self.cov3d_91_(output)
        output = self.drop_91(output)
        output = self.bn_91(output)
        output = self.relu_91(output)
        output = self.cov3d_92(output)
        output = self.drop_92(output)
        output = self.bn_92(output)
        output9 = self.relu_92(output)

        # 8 * 32 * 32
        output = self.t_cov3d_10(output9)
        output = torch.cat((output, output3), dim = 1)
        output = self.cov3d_101(output)
        output = self.drop_101(output)
        output = self.bn_101(output)
        output = self.relu_101(output)
        output = self.cov3d_102(output)
        output = self.drop_102(output)
        output = self.bn_102(output)
        output10 = self.relu_102(output)

        # 16 * 64 * 64
        output = self.t_cov3d_11(output10)
        output = torch.cat((output, output2), dim = 1)
        output = self.cov3d_111(output)
        output = self.drop_111(output)
        output = self.bn_111(output)
        output = self.relu_111(output)
        output = self.cov3d_112(output)
        output = self.drop_112(output)
        output = self.bn_112(output)
        output11 = self.relu_112(output)

        # 32 * 128 * 128
        output = self.t_cov3d_12(output11)
        output = torch.cat((output, output1), dim = 1)
        output = self.cov3d_121(output)
        output = self.drop_121(output)
        output = self.bn_121(output)
        output = self.relu_121(output)
        output = self.cov3d_122(output)
        output = self.drop_122(output)
        output = self.bn_122(output)
        output12 = self.relu_122(output)

        output = self.cov3d_13(output12)

        return output


class UNet_3D_At(Module):
    """
    Model 1
    """
    def __init__(self):
        super(UNet_3D_At, self).__init__()
        # 32 * 128 * 128
        self.cov3d_11 = nn.Conv3d(in_channels = 1, out_channels = 16, kernel_size = 3, padding = 2, dilation = 2)
        self.bn_11 = nn.BatchNorm3d(num_features = 16)
        self.relu_11 = nn.ReLU(inplace = False)
        self.cov3d_12 = nn.Conv3d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = 2, dilation = 2)
        self.bn_12 = nn.BatchNorm3d(num_features = 16)
        self.relu_12 = nn.ReLU(inplace = False)
        self.maxpool_1 = nn.MaxPool3d(kernel_size = 2)

        # 16 * 64 * 64
        self.cov3d_21 = nn.Conv3d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 2, dilation = 2)
        self.bn_21 = nn.BatchNorm3d(num_features = 32)
        self.relu_21 = nn.ReLU(inplace = False)
        self.cov3d_22 = nn.Conv3d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 2, dilation = 2)
        self.bn_22 = nn.BatchNorm3d(num_features = 32)
        self.relu_22 = nn.ReLU(inplace = False)
        self.maxpool_2 = nn.MaxPool3d(kernel_size = 2)

        # 8 * 32 * 32
        self.cov3d_31 = nn.Conv3d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 2, dilation = 2)
        self.bn_31 = nn.BatchNorm3d(num_features = 64)
        self.relu_31 = nn.ReLU(inplace = False)
        self.cov3d_32 = nn.Conv3d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 2, dilation = 2)
        self.bn_32 = nn.BatchNorm3d(num_features = 64)
        self.relu_32 = nn.ReLU(inplace = False)
        self.maxpool_3 = nn.MaxPool3d(kernel_size = 2)

        #  4 * 16 * 16
        self.cov3d_41 = nn.Conv3d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 2, dilation = 2)
        self.bn_41 = nn.BatchNorm3d(num_features = 128)
        self.relu_41 = nn.ReLU(inplace = False)
        self.cov3d_42 = nn.Conv3d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 2, dilation = 2)
        self.bn_42 = nn.BatchNorm3d(num_features = 128)
        self.relu_42 = nn.ReLU(inplace = False)

        self.maxpool_4 = nn.MaxPool3d(kernel_size = 2)

        self.con3d_51 = nn.Conv3d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1)
        self.bn_51 = nn.BatchNorm3d(num_features = 256)
        self.relu_51 = nn.ReLU(inplace = False)
        self.con3d_52 = nn.Conv3d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        self.drop_52 = nn.Dropout3d(p = 0.5)
        self.bn_52 = nn.BatchNorm3d(num_features = 256)
        self.relu_52 = nn.ReLU(inplace = False)

        # 4 * 16 * 16
        self.t_cov3d_9 =nn.ConvTranspose3d(in_channels = 256, out_channels = 128, kernel_size = 3, stride = 2, output_padding = 1, padding = 1)
        self.cov3d_91_ = nn.Conv3d(in_channels = 256, out_channels = 128, kernel_size = 3, padding = 1)
        self.bn_91 = nn.BatchNorm3d(num_features = 128)
        self.relu_91 = nn.ReLU(inplace = False)
        self.cov3d_92 = nn.Conv3d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
        self.bn_92 = nn.BatchNorm3d(num_features = 128)
        self.relu_92 = nn.ReLU(inplace = False)

        # 8 * 32 * 32
        self.t_cov3d_10 = nn.ConvTranspose3d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 2, output_padding = 1, padding = 1)
        self.cov3d_101 = nn.Conv3d(in_channels = 128, out_channels = 64, kernel_size = 3, padding = 1)
        self.bn_101 = nn.BatchNorm3d(num_features = 64)
        self.relu_101 = nn.ReLU(inplace = False)
        self.cov3d_102 = nn.Conv3d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
        self.bn_102 = nn.BatchNorm3d(num_features = 64)
        self.relu_102 = nn.ReLU(inplace = False)

        # 16 * 64 * 64
        self.t_cov3d_11 = nn.ConvTranspose3d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 2, output_padding = 1, padding = 1)
        self.cov3d_111 = nn.Conv3d(in_channels = 64, out_channels = 32, kernel_size = 3, padding = 1)
        self.bn_111 = nn.BatchNorm3d(num_features = 32)
        self.relu_111 = nn.ReLU(inplace = False)
        self.drop_112 = nn.Dropout3d(p = 0.0)
        self.bn_112 = nn.BatchNorm3d(num_features = 32)
        self.relu_112 = nn.ReLU(inplace = False)

        # 32 * 128 * 128
        self.t_cov3d_12 = nn.ConvTranspose3d(in_channels = 32, out_channels = 16, kernel_size = 3, stride = 2, output_padding = 1, padding = 1)
        self.cov3d_121 = nn.Conv3d(in_channels = 32, out_channels = 16, kernel_size = 3, padding = 1)
        self.bn_121 = nn.BatchNorm3d(num_features = 16)
        self.relu_121 = nn.ReLU(inplace = False)
        self.cov3d_122 = nn.Conv3d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = 1)
        self.bn_122 = nn.BatchNorm3d(num_features = 16)
        self.relu_122 = nn.ReLU(inplace = False)

        # 128 * 128
        self.cov3d_13 = nn.Conv3d(in_channels = 16, out_channels = 3, kernel_size = 1)

    def forward(self, input):
        # 32 * 128 * 128
        output = self.cov3d_11(input)
        #output = self.drop_11(output)
        output = self.bn_11(output)
        output = self.relu_11(output)
        output = self.cov3d_12(output)
        #output = self.drop_12(output)
        output = self.bn_12(output)
        output1 = self.relu_12(output)
        output = self.maxpool_1(output1)

        # 16 * 64 * 64
        output = self.cov3d_21(output)
        #output = self.drop_21(output)
        output = self.bn_21(output)
        output = self.relu_21(output)
        output = self.cov3d_22(output)
        #output = self.drop_22(output)
        output = self.bn_22(output)
        output2 = self.relu_22(output)
        output = self.maxpool_2(output2)

        # 8 * 32 * 32
        output = self.cov3d_31(output)
        #output = self.drop_31(output)
        output = self.bn_31(output)
        output = self.relu_31(output)
        output = self.cov3d_32(output)
        #output = self.drop_32(output)
        output = self.bn_32(output)
        output3 = self.relu_32(output)
        output = self.maxpool_3(output3)

        # 4 * 16 * 16
        output = self.cov3d_41(output)
        #output = self.drop_41(output)
        output = self.bn_41(output)
        output = self.relu_41(output)
        output = self.cov3d_42(output)
        #output = self.drop_42(output)
        output = self.bn_42(output)
        output4 = self.relu_42(output)
        output = self.maxpool_4(output4)

        output = self.con3d_51(output)
        #output = self.drop_51(output)
        output = self.bn_51(output)
        output = self.relu_51(output)
        output = self.con3d_52(output)
        output = self.drop_52(output)
        output = self.bn_52(output)
        output5 = self.relu_52(output)

        # 4 * 16 * 16
        output = self.t_cov3d_9(output5)
        output = torch.cat((output, output4), dim = 1)
        output = self.cov3d_91_(output)
        #output = self.drop_91(output)
        output = self.bn_91(output)
        output = self.relu_91(output)
        output = self.cov3d_92(output)
        #output = self.drop_92(output)
        output = self.bn_92(output)
        output9 = self.relu_92(output)

        # 8 * 32 * 32
        output = self.t_cov3d_10(output9)
        output = torch.cat((output, output3), dim = 1)
        output = self.cov3d_101(output)
        #output = self.drop_101(output)
        output = self.bn_101(output)
        output = self.relu_101(output)
        output = self.cov3d_102(output)
        #output = self.drop_102(output)
        output = self.bn_102(output)
        output10 = self.relu_102(output)

        # 16 * 64 * 64
        output = self.t_cov3d_11(output10)
        output = torch.cat((output, output2), dim = 1)
        output = self.cov3d_111(output)
        #output = self.drop_111(output)
        output = self.bn_111(output)
        output = self.relu_111(output)
        output = self.cov3d_112(output)
        #output = self.drop_112(output)
        output = self.bn_112(output)
        output11 = self.relu_112(output)

        # 32 * 128 * 128
        output = self.t_cov3d_12(output11)
        output = torch.cat((output, output1), dim = 1)
        output = self.cov3d_121(output)
        #output = self.drop_121(output)
        output = self.bn_121(output)
        output = self.relu_121(output)
        output = self.cov3d_122(output)
        #output = self.drop_122(output)
        output = self.bn_122(output)
        output12 = self.relu_122(output)

        output = self.cov3d_13(output12)

        return output


class UNet_3D_D_At(Module):
    """
    Model 4
    """
    def __init__(self):
        super(UNet_3D_D_At, self).__init__()
        # 32 * 128 * 128
        self.cov3d_11 = nn.Conv3d(in_channels = 1, out_channels = 16, kernel_size = 3, padding = 2, dilation = 2)
        self.drop_11 = nn.Dropout3d(p = 0.07)
        self.bn_11 = nn.BatchNorm3d(num_features = 16)
        self.relu_11 = nn.ReLU(inplace = False)
        self.cov3d_12 = nn.Conv3d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = 2, dilation = 2)
        self.drop_12 = nn.Dropout3d(p = 0.07)
        self.bn_12 = nn.BatchNorm3d(num_features = 16)
        self.relu_12 = nn.ReLU(inplace = False)
        self.maxpool_1 = nn.MaxPool3d(kernel_size = 2)

        # 16 * 64 * 64
        self.cov3d_21 = nn.Conv3d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 2, dilation = 2)
        self.drop_21 = nn.Dropout3d(p = 0.1)
        self.bn_21 = nn.BatchNorm3d(num_features = 32)
        self.relu_21 = nn.ReLU(inplace = False)
        self.cov3d_22 = nn.Conv3d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 2, dilation = 2)
        self.drop_22 = nn.Dropout3d(p = 0.1)
        self.bn_22 = nn.BatchNorm3d(num_features = 32)
        self.relu_22 = nn.ReLU(inplace = False)
        self.maxpool_2 = nn.MaxPool3d(kernel_size = 2)

        # 8 * 32 * 32
        self.cov3d_31 = nn.Conv3d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 2, dilation = 2)
        self.drop_31 = nn.Dropout3d(p = 0.1)
        self.bn_31 = nn.BatchNorm3d(num_features = 64)
        self.relu_31 = nn.ReLU(inplace = False)
        self.cov3d_32 = nn.Conv3d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 2, dilation = 2)
        self.drop_32 = nn.Dropout3d(p = 0.1)
        self.bn_32 = nn.BatchNorm3d(num_features = 64)
        self.relu_32 = nn.ReLU(inplace = False)
        self.maxpool_3 = nn.MaxPool3d(kernel_size = 2)

        #  4 * 16 * 16
        self.cov3d_41 = nn.Conv3d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 2, dilation = 2)
        self.drop_41 = nn.Dropout3d(p = 0.125)
        self.bn_41 = nn.BatchNorm3d(num_features = 128)
        self.relu_41 = nn.ReLU(inplace = False)
        self.cov3d_42 = nn.Conv3d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 2, dilation = 2)
        self.drop_42 = nn.Dropout3d(p = 0.125)
        self.bn_42 = nn.BatchNorm3d(num_features = 128)
        self.relu_42 = nn.ReLU(inplace = False)

        self.maxpool_4 = nn.MaxPool3d(kernel_size = 2)

        self.con3d_51 = nn.Conv3d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1)
        self.drop_51 = nn.Dropout3d(p = 0.25)
        self.bn_51 = nn.BatchNorm3d(num_features = 256)
        self.relu_51 = nn.ReLU(inplace = False)
        self.con3d_52 = nn.Conv3d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        self.drop_52 = nn.Dropout3d(p = 0.25)
        self.bn_52 = nn.BatchNorm3d(num_features = 256)
        self.relu_52 = nn.ReLU(inplace = False)

        # 4 * 16 * 16
        self.t_cov3d_9 =nn.ConvTranspose3d(in_channels = 256, out_channels = 128, kernel_size = 3, stride = 2, output_padding = 1, padding = 1)
        self.cov3d_91_ = nn.Conv3d(in_channels = 256, out_channels = 128, kernel_size = 3, padding = 1)
        self.drop_91 = nn.Dropout3d(p = 0.125)
        self.bn_91 = nn.BatchNorm3d(num_features = 128)
        self.relu_91 = nn.ReLU(inplace = False)
        self.cov3d_92 = nn.Conv3d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
        self.drop_92 = nn.Dropout3d(p = 0.125)
        self.bn_92 = nn.BatchNorm3d(num_features = 128)
        self.relu_92 = nn.ReLU(inplace = False)

        # 8 * 32 * 32
        self.t_cov3d_10 = nn.ConvTranspose3d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 2, output_padding = 1, padding = 1)
        self.cov3d_101 = nn.Conv3d(in_channels = 128, out_channels = 64, kernel_size = 3, padding = 1)
        self.drop_101 = nn.Dropout3d(p = 0.1)
        self.bn_101 = nn.BatchNorm3d(num_features = 64)
        self.relu_101 = nn.ReLU(inplace = False)
        self.cov3d_102 = nn.Conv3d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
        self.drop_102 = nn.Dropout3d(p = 0.1)
        self.bn_102 = nn.BatchNorm3d(num_features = 64)
        self.relu_102 = nn.ReLU(inplace = False)

        # 16 * 64 * 64
        self.t_cov3d_11 = nn.ConvTranspose3d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 2, output_padding = 1, padding = 1)
        self.cov3d_111 = nn.Conv3d(in_channels = 64, out_channels = 32, kernel_size = 3, padding = 1)
        self.drop_111 = nn.Dropout3d(p = 0.1)
        self.bn_111 = nn.BatchNorm3d(num_features = 32)
        self.relu_111 = nn.ReLU(inplace = False)
        self.cov3d_112 = nn.Conv3d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1)
        self.drop_112 = nn.Dropout3d(p = 0.1)
        self.bn_112 = nn.BatchNorm3d(num_features = 32)
        self.relu_112 = nn.ReLU(inplace = False)

        # 32 * 128 * 128
        self.t_cov3d_12 = nn.ConvTranspose3d(in_channels = 32, out_channels = 16, kernel_size = 3, stride = 2, output_padding = 1, padding = 1)
        self.cov3d_121 = nn.Conv3d(in_channels = 32, out_channels = 16, kernel_size = 3, padding = 1)
        self.drop_121 = nn.Dropout3d(p = 0.07)
        self.bn_121 = nn.BatchNorm3d(num_features = 16)
        self.relu_121 = nn.ReLU(inplace = False)
        self.cov3d_122 = nn.Conv3d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = 1)
        self.drop_122 = nn.Dropout3d(p = 0.07)
        self.bn_122 = nn.BatchNorm3d(num_features = 16)
        self.relu_122 = nn.ReLU(inplace = False)

        # 128 * 128
        self.cov3d_13 = nn.Conv3d(in_channels = 16, out_channels = 3, kernel_size = 1)

    def forward(self, input):
        # 32 * 128 * 128
        output = self.cov3d_11(input)
        output = self.drop_11(output)
        output = self.bn_11(output)
        output = self.relu_11(output)
        output = self.cov3d_12(output)
        output = self.drop_12(output)
        output = self.bn_12(output)
        output1 = self.relu_12(output)
        output = self.maxpool_1(output1)

        # 16 * 64 * 64
        output = self.cov3d_21(output)
        output = self.drop_21(output)
        output = self.bn_21(output)
        output = self.relu_21(output)
        output = self.cov3d_22(output)
        output = self.drop_22(output)
        output = self.bn_22(output)
        output2 = self.relu_22(output)
        output = self.maxpool_2(output2)

        # 8 * 32 * 32
        output = self.cov3d_31(output)
        output = self.drop_31(output)
        output = self.bn_31(output)
        output = self.relu_31(output)
        output = self.cov3d_32(output)
        output = self.drop_32(output)
        output = self.bn_32(output)
        output3 = self.relu_32(output)
        output = self.maxpool_3(output3)

        # 4 * 16 * 16
        output = self.cov3d_41(output)
        output = self.drop_41(output)
        output = self.bn_41(output)
        output = self.relu_41(output)
        output = self.cov3d_42(output)
        output = self.drop_42(output)
        output = self.bn_42(output)
        output4 = self.relu_42(output)
        output = self.maxpool_4(output4)

        output = self.con3d_51(output)
        output = self.drop_51(output)
        output = self.bn_51(output)
        output = self.relu_51(output)
        output = self.con3d_52(output)
        output = self.drop_52(output)
        output = self.bn_52(output)
        output5 = self.relu_52(output)

        # 4 * 16 * 16
        output = self.t_cov3d_9(output5)
        output = torch.cat((output, output4), dim = 1)
        output = self.cov3d_91_(output)
        output = self.drop_91(output)
        output = self.bn_91(output)
        output = self.relu_91(output)
        output = self.cov3d_92(output)
        output = self.drop_92(output)
        output = self.bn_92(output)
        output9 = self.relu_92(output)

        # 8 * 32 * 32
        output = self.t_cov3d_10(output9)
        output = torch.cat((output, output3), dim = 1)
        output = self.cov3d_101(output)
        output = self.drop_101(output)
        output = self.bn_101(output)
        output = self.relu_101(output)
        output = self.cov3d_102(output)
        output = self.drop_102(output)
        output = self.bn_102(output)
        output10 = self.relu_102(output)

        # 16 * 64 * 64
        output = self.t_cov3d_11(output10)
        output = torch.cat((output, output2), dim = 1)
        output = self.cov3d_111(output)
        output = self.drop_111(output)
        output = self.bn_111(output)
        output = self.relu_111(output)
        output = self.cov3d_112(output)
        output = self.drop_112(output)
        output = self.bn_112(output)
        output11 = self.relu_112(output)

        # 32 * 128 * 128
        output = self.t_cov3d_12(output11)
        output = torch.cat((output, output1), dim = 1)
        output = self.cov3d_121(output)
        output = self.drop_121(output)
        output = self.bn_121(output)
        output = self.relu_121(output)
        output = self.cov3d_122(output)
        output = self.drop_122(output)
        output = self.bn_122(output)
        output12 = self.relu_122(output)

        output = self.cov3d_13(output12)

        return output


class UNet_3D_At_As(Module):
    """
    Model 2
    """
    def __init__(self):
        super(UNet_3D_At_As, self).__init__()
        # 32 * 128 * 128
        self.cov3d_11 = nn.Conv3d(in_channels = 1, out_channels = 16, kernel_size = 3, padding = 1, dilation = 1)
        self.drop_11 = nn.Dropout3d(p = 0.0)
        self.bn_11 = nn.BatchNorm3d(num_features = 16)
        self.relu_11 = nn.ReLU()
        self.cov3d_12 = nn.Conv3d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = 2, dilation = 2)
        self.drop_12 = nn.Dropout3d(p = 0.0)
        self.bn_12 = nn.BatchNorm3d(num_features = 16)
        self.relu_12 = nn.ReLU()
        self.maxpool_1 = nn.MaxPool3d(kernel_size = 2)

        # 16 * 64 * 64
        self.cov3d_21 = nn.Conv3d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1, dilation = 1)
        self.drop_21 = nn.Dropout3d(p = 0.0)
        self.bn_21 = nn.BatchNorm3d(num_features = 32)
        self.relu_21 = nn.ReLU()
        self.cov3d_22 = nn.Conv3d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 2, dilation = 2)
        self.drop_22 = nn.Dropout3d(p = 0.0)
        self.bn_22 = nn.BatchNorm3d(num_features = 32)
        self.relu_22 = nn.ReLU()
        self.maxpool_2 = nn.MaxPool3d(kernel_size = 2)

        # 8 * 32 * 32
        self.cov3d_31 = nn.Conv3d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1, dilation = 1)
        self.drop_31 = nn.Dropout3d(p = 0.0)
        self.bn_31 = nn.BatchNorm3d(num_features = 64)
        self.relu_31 = nn.ReLU()
        self.cov3d_32 = nn.Conv3d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 2, dilation = 2)
        self.drop_32 = nn.Dropout3d(p = 0.0)
        self.bn_32 = nn.BatchNorm3d(num_features = 64)
        self.relu_32 = nn.ReLU()
        self.maxpool_3 = nn.MaxPool3d(kernel_size = 2)

        #  4 * 16 * 16
        self.cov3d_41 = nn.Conv3d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1, dilation = 1)
        self.drop_41 = nn.Dropout3d(p = 0.0)
        self.bn_41 = nn.BatchNorm3d(num_features = 128)
        self.relu_41 = nn.ReLU()
        self.cov3d_42 = nn.Conv3d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 2, dilation = 2)
        self.drop_42 = nn.Dropout3d(p = 0.0)
        self.bn_42 = nn.BatchNorm3d(num_features = 128)
        self.relu_42 = nn.ReLU()

        self.con3d_r0 = nn.Conv3d(in_channels = 128, out_channels = 16, kernel_size = 1, padding = 0)
        self.con3d_r1 = nn.Conv3d(in_channels = 128, out_channels = 16, kernel_size = 3, padding = 1, dilation = 1)
        self.con3d_r2 = nn.Conv3d(in_channels = 128, out_channels = 16, kernel_size = 3, padding = 2, dilation = 2)
        self.con3d_r4 = nn.Conv3d(in_channels = 128, out_channels = 16, kernel_size = 3, padding = 4, dilation = 4)
        self.con3d_r6 = nn.Conv3d(in_channels = 128, out_channels = 16, kernel_size = 3, padding = 6, dilation = 6)
        self.con3d_r8 = nn.Conv3d(in_channels = 128, out_channels = 16, kernel_size = 3, padding = 8, dilation = 8)
        self.con3d_input = nn.Conv3d(in_channels = 1, out_channels = 16, kernel_size = 3, padding = 1, stride = 8)

        self.con3d_5 = nn.Conv3d(in_channels = 112, out_channels = 128, kernel_size = 3, padding = 1)
        self.drop_5 = nn.Dropout3d(p = 0.0)
        self.bn_5 = nn.BatchNorm3d(num_features = 128)
        self.relu_5 = nn.ReLU()

        # 4 * 16 * 16
        self.cov3d_91_ = nn.Conv3d(in_channels = 256, out_channels = 128, kernel_size = 3, padding = 1)
        self.drop_91 = nn.Dropout3d(p = 0.0)
        self.bn_91 = nn.BatchNorm3d(num_features = 128)
        self.relu_91 = nn.ReLU()
        self.cov3d_92 = nn.Conv3d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
        self.drop_92 = nn.Dropout3d(p = 0.0)
        self.bn_92 = nn.BatchNorm3d(num_features = 128)
        self.relu_92 = nn.ReLU()

        # 8 * 32 * 32
        self.t_cov3d_10 = nn.ConvTranspose3d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 2, output_padding = 1, padding = 1)
        self.cov3d_101 = nn.Conv3d(in_channels = 128, out_channels = 64, kernel_size = 3, padding = 1)
        self.drop_101 = nn.Dropout3d(p = 0.0)
        self.bn_101 = nn.BatchNorm3d(num_features = 64)
        self.relu_101 = nn.ReLU()
        self.cov3d_102 = nn.Conv3d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
        self.drop_102 = nn.Dropout3d(p = 0.0)
        self.bn_102 = nn.BatchNorm3d(num_features = 64)
        self.relu_102 = nn.ReLU()

        # 16 * 64 * 64
        self.t_cov3d_11 = nn.ConvTranspose3d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 2, output_padding = 1, padding = 1)
        self.cov3d_111 = nn.Conv3d(in_channels = 64, out_channels = 32, kernel_size = 3, padding = 1)
        self.drop_111 = nn.Dropout3d(p = 0.0)
        self.bn_111 = nn.BatchNorm3d(num_features = 32)
        self.relu_111 = nn.ReLU()
        self.cov3d_112 = nn.Conv3d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1)
        self.drop_112 = nn.Dropout3d(p = 0.0)
        self.bn_112 = nn.BatchNorm3d(num_features = 32)
        self.relu_112 = nn.ReLU()

        # 32 * 128 * 128
        self.t_cov3d_12 = nn.ConvTranspose3d(in_channels = 32, out_channels = 16, kernel_size = 3, stride = 2, output_padding = 1, padding = 1)
        self.cov3d_121 = nn.Conv3d(in_channels = 32, out_channels = 16, kernel_size = 3, padding = 1)
        self.drop_121 = nn.Dropout3d(p = 0.0)
        self.bn_121 = nn.BatchNorm3d(num_features = 16)
        self.relu_121 = nn.ReLU()
        self.cov3d_122 = nn.Conv3d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = 1)
        self.drop_122 = nn.Dropout3d(p = 0.0)
        self.bn_122 = nn.BatchNorm3d(num_features = 16)
        self.relu_122 = nn.ReLU()

        # 128 * 128
        self.cov3d_13 = nn.Conv3d(in_channels = 16, out_channels = 3, kernel_size = 1)

    def forward(self, input):
        # 32 * 128 * 128
        output = self.cov3d_11(input)
        output = self.drop_11(output)
        output = self.bn_11(output)
        output = self.relu_11(output)
        output = self.cov3d_12(output)
        output = self.drop_12(output)
        output = self.bn_12(output)
        output1 = self.relu_12(output)
        output = self.maxpool_1(output1)

        # 16 * 64 * 64
        output = self.cov3d_21(output)
        output = self.drop_21(output)
        output = self.bn_21(output)
        output = self.relu_21(output)
        output = self.cov3d_22(output)
        output = self.drop_22(output)
        output = self.bn_22(output)
        output2 = self.relu_22(output)
        output = self.maxpool_2(output2)

        # 8 * 32 * 32
        output = self.cov3d_31(output)
        output = self.drop_31(output)
        output = self.bn_31(output)
        output = self.relu_31(output)
        output = self.cov3d_32(output)
        output = self.drop_32(output)
        output = self.bn_32(output)
        output3 = self.relu_32(output)
        output = self.maxpool_3(output3)

        # 4 * 16 * 16
        output = self.cov3d_41(output)
        output = self.drop_41(output)
        output = self.bn_41(output)
        output = self.relu_41(output)
        output = self.cov3d_42(output)
        output = self.drop_42(output)
        output = self.bn_42(output)
        output4 = self.relu_42(output)

        output_r0 = self.con3d_r0(output4)
        output_r1 = self.con3d_r1(output4)
        output_r2 = self.con3d_r2(output4)
        output_r4 = self.con3d_r4(output4)
        output_r6 = self.con3d_r6(output4)
        output_r8 = self.con3d_r8(output4)
        output_r9 = self.con3d_input(input)

        output = torch.cat((output_r0, output_r1, output_r2, output_r4, output_r6, output_r8, output_r9), dim = 1)
        output = self.con3d_5(output)
        output = self.drop_5(output)
        output = self.bn_5(output)
        output = self.relu_5(output)

        # 4 * 16 * 16
        output = torch.cat((output, output4), dim = 1)
        output = self.cov3d_91_(output)
        output = self.drop_91(output)
        output = self.bn_91(output)
        output = self.relu_91(output)
        output = self.cov3d_92(output)
        output = self.drop_92(output)
        output = self.bn_92(output)
        output9 = self.relu_92(output)

        # 8 * 32 * 32
        output = self.t_cov3d_10(output9)
        output = torch.cat((output, output3), dim = 1)
        output = self.cov3d_101(output)
        output = self.drop_101(output)
        output = self.bn_101(output)
        output = self.relu_101(output)
        output = self.cov3d_102(output)
        output = self.drop_102(output)
        output = self.bn_102(output)
        output10 = self.relu_102(output)

        # 16 * 64 * 64
        output = self.t_cov3d_11(output10)
        output = torch.cat((output, output2), dim = 1)
        output = self.cov3d_111(output)
        output = self.drop_111(output)
        output = self.bn_111(output)
        output = self.relu_111(output)
        output = self.cov3d_112(output)
        output = self.drop_112(output)
        output = self.bn_112(output)
        output11 = self.relu_112(output)

        # 32 * 128 * 128
        output = self.t_cov3d_12(output11)
        output = torch.cat((output, output1), dim = 1)
        output = self.cov3d_121(output)
        output = self.drop_121(output)
        output = self.bn_121(output)
        output = self.relu_121(output)
        output = self.cov3d_122(output)
        output = self.drop_122(output)
        output = self.bn_122(output)
        output12 = self.relu_122(output)

        output = self.cov3d_13(output12)

        return output


class UNet_3D_D_At_As(Module):
    """
    Model 5
    """
    def __init__(self):
        super(UNet_3D_D_At_As, self).__init__()
        # 32 * 128 * 128
        self.cov3d_11 = nn.Conv3d(in_channels = 1, out_channels = 16, kernel_size = 3, padding = 1, dilation = 1)
        self.drop_11 = nn.Dropout3d(p = 0.1)
        self.bn_11 = nn.BatchNorm3d(num_features = 16)
        self.relu_11 = nn.ReLU()
        self.cov3d_12 = nn.Conv3d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = 2, dilation = 2)
        self.drop_12 = nn.Dropout3d(p = 0.1)
        self.bn_12 = nn.BatchNorm3d(num_features = 16)
        self.relu_12 = nn.ReLU()
        self.maxpool_1 = nn.MaxPool3d(kernel_size = 2)

        # 16 * 64 * 64
        self.cov3d_21 = nn.Conv3d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1, dilation = 1)
        self.drop_21 = nn.Dropout3d(p = 0.1)
        self.bn_21 = nn.BatchNorm3d(num_features = 32)
        self.relu_21 = nn.ReLU()
        self.cov3d_22 = nn.Conv3d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 2, dilation = 2)
        self.drop_22 = nn.Dropout3d(p = 0.1)
        self.bn_22 = nn.BatchNorm3d(num_features = 32)
        self.relu_22 = nn.ReLU()
        self.maxpool_2 = nn.MaxPool3d(kernel_size = 2)

        # 8 * 32 * 32
        self.cov3d_31 = nn.Conv3d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1, dilation = 1)
        self.drop_31 = nn.Dropout3d(p = 0.15)
        self.bn_31 = nn.BatchNorm3d(num_features = 64)
        self.relu_31 = nn.ReLU()
        self.cov3d_32 = nn.Conv3d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 2, dilation = 2)
        self.drop_32 = nn.Dropout3d(p = 0.15)
        self.bn_32 = nn.BatchNorm3d(num_features = 64)
        self.relu_32 = nn.ReLU()
        self.maxpool_3 = nn.MaxPool3d(kernel_size = 2)

        #  4 * 16 * 16
        self.cov3d_41 = nn.Conv3d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1, dilation = 1)
        self.drop_41 = nn.Dropout3d(p = 0.2)
        self.bn_41 = nn.BatchNorm3d(num_features = 128)
        self.relu_41 = nn.ReLU()
        self.cov3d_42 = nn.Conv3d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 2, dilation = 2)
        self.drop_42 = nn.Dropout3d(p = 0.2)
        self.bn_42 = nn.BatchNorm3d(num_features = 128)
        self.relu_42 = nn.ReLU()

        self.con3d_r0 = nn.Conv3d(in_channels = 128, out_channels = 16, kernel_size = 1, padding = 0)
        self.con3d_r1 = nn.Conv3d(in_channels = 128, out_channels = 16, kernel_size = 3, padding = 1, dilation = 1)
        self.con3d_r2 = nn.Conv3d(in_channels = 128, out_channels = 16, kernel_size = 3, padding = 2, dilation = 2)
        self.con3d_r4 = nn.Conv3d(in_channels = 128, out_channels = 16, kernel_size = 3, padding = 4, dilation = 4)
        self.con3d_r6 = nn.Conv3d(in_channels = 128, out_channels = 16, kernel_size = 3, padding = 6, dilation = 6)
        self.con3d_r8 = nn.Conv3d(in_channels = 128, out_channels = 16, kernel_size = 3, padding = 8, dilation = 8)
        self.con3d_input = nn.Conv3d(in_channels = 1, out_channels = 16, kernel_size = 3, padding = 1, stride = 8)

        self.con3d_5 = nn.Conv3d(in_channels = 112, out_channels = 128, kernel_size = 3, padding = 1)
        self.drop_5 = nn.Dropout3d(p = 0.25)
        self.bn_5 = nn.BatchNorm3d(num_features = 128)
        self.relu_5 = nn.ReLU()

        # 4 * 16 * 16
        self.cov3d_91_ = nn.Conv3d(in_channels = 256, out_channels = 128, kernel_size = 3, padding = 1)
        self.drop_91 = nn.Dropout3d(p = 0.2)
        self.bn_91 = nn.BatchNorm3d(num_features = 128)
        self.relu_91 = nn.ReLU()
        self.cov3d_92 = nn.Conv3d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
        self.drop_92 = nn.Dropout3d(p = 0.2)
        self.bn_92 = nn.BatchNorm3d(num_features = 128)
        self.relu_92 = nn.ReLU()

        # 8 * 32 * 32
        self.t_cov3d_10 = nn.ConvTranspose3d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 2, output_padding = 1, padding = 1)
        self.cov3d_101 = nn.Conv3d(in_channels = 128, out_channels = 64, kernel_size = 3, padding = 1)
        self.drop_101 = nn.Dropout3d(p = 0.15)
        self.bn_101 = nn.BatchNorm3d(num_features = 64)
        self.relu_101 = nn.ReLU()
        self.cov3d_102 = nn.Conv3d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
        self.drop_102 = nn.Dropout3d(p = 0.15)
        self.bn_102 = nn.BatchNorm3d(num_features = 64)
        self.relu_102 = nn.ReLU()

        # 16 * 64 * 64
        self.t_cov3d_11 = nn.ConvTranspose3d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 2, output_padding = 1, padding = 1)
        self.cov3d_111 = nn.Conv3d(in_channels = 64, out_channels = 32, kernel_size = 3, padding = 1)
        self.drop_111 = nn.Dropout3d(p = 0.1)
        self.bn_111 = nn.BatchNorm3d(num_features = 32)
        self.relu_111 = nn.ReLU()
        self.cov3d_112 = nn.Conv3d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1)
        self.drop_112 = nn.Dropout3d(p = 0.1)
        self.bn_112 = nn.BatchNorm3d(num_features = 32)
        self.relu_112 = nn.ReLU()

        # 32 * 128 * 128
        self.t_cov3d_12 = nn.ConvTranspose3d(in_channels = 32, out_channels = 16, kernel_size = 3, stride = 2, output_padding = 1, padding = 1)
        self.cov3d_121 = nn.Conv3d(in_channels = 32, out_channels = 16, kernel_size = 3, padding = 1)
        self.drop_121 = nn.Dropout3d(p = 0.1)
        self.bn_121 = nn.BatchNorm3d(num_features = 16)
        self.relu_121 = nn.ReLU()
        self.cov3d_122 = nn.Conv3d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = 1)
        self.drop_122 = nn.Dropout3d(p = 0.1)
        self.bn_122 = nn.BatchNorm3d(num_features = 16)
        self.relu_122 = nn.ReLU()

        # 128 * 128
        self.cov3d_13 = nn.Conv3d(in_channels = 16, out_channels = 3, kernel_size = 1)

    def forward(self, input):
        # 32 * 128 * 128
        output = self.cov3d_11(input)
        output = self.drop_11(output)
        output = self.bn_11(output)
        output = self.relu_11(output)
        output = self.cov3d_12(output)
        output = self.drop_12(output)
        output = self.bn_12(output)
        output1 = self.relu_12(output)
        output = self.maxpool_1(output1)

        # 16 * 64 * 64
        output = self.cov3d_21(output)
        output = self.drop_21(output)
        output = self.bn_21(output)
        output = self.relu_21(output)
        output = self.cov3d_22(output)
        output = self.drop_22(output)
        output = self.bn_22(output)
        output2 = self.relu_22(output)
        output = self.maxpool_2(output2)

        # 8 * 32 * 32
        output = self.cov3d_31(output)
        output = self.drop_31(output)
        output = self.bn_31(output)
        output = self.relu_31(output)
        output = self.cov3d_32(output)
        output = self.drop_32(output)
        output = self.bn_32(output)
        output3 = self.relu_32(output)
        output = self.maxpool_3(output3)

        # 4 * 16 * 16
        output = self.cov3d_41(output)
        output = self.drop_41(output)
        output = self.bn_41(output)
        output = self.relu_41(output)
        output = self.cov3d_42(output)
        output = self.drop_42(output)
        output = self.bn_42(output)
        output4 = self.relu_42(output)

        output_r0 = self.con3d_r0(output4)
        output_r1 = self.con3d_r1(output4)
        output_r2 = self.con3d_r2(output4)
        output_r4 = self.con3d_r4(output4)
        output_r6 = self.con3d_r6(output4)
        output_r8 = self.con3d_r8(output4)
        output_r9 = self.con3d_input(input)

        output = torch.cat((output_r0, output_r1, output_r2, output_r4, output_r6, output_r8, output_r9), dim = 1)
        output = self.con3d_5(output)
        output = self.drop_5(output)
        output = self.bn_5(output)
        output = self.relu_5(output)

        # 4 * 16 * 16
        output = torch.cat((output, output4), dim = 1)
        output = self.cov3d_91_(output)
        output = self.drop_91(output)
        output = self.bn_91(output)
        output = self.relu_91(output)
        output = self.cov3d_92(output)
        output = self.drop_92(output)
        output = self.bn_92(output)
        output9 = self.relu_92(output)

        # 8 * 32 * 32
        output = self.t_cov3d_10(output9)
        output = torch.cat((output, output3), dim = 1)
        output = self.cov3d_101(output)
        output = self.drop_101(output)
        output = self.bn_101(output)
        output = self.relu_101(output)
        output = self.cov3d_102(output)
        output = self.drop_102(output)
        output = self.bn_102(output)
        output10 = self.relu_102(output)

        # 16 * 64 * 64
        output = self.t_cov3d_11(output10)
        output = torch.cat((output, output2), dim = 1)
        output = self.cov3d_111(output)
        output = self.drop_111(output)
        output = self.bn_111(output)
        output = self.relu_111(output)
        output = self.cov3d_112(output)
        output = self.drop_112(output)
        output = self.bn_112(output)
        output11 = self.relu_112(output)

        # 32 * 128 * 128
        output = self.t_cov3d_12(output11)
        output = torch.cat((output, output1), dim = 1)
        output = self.cov3d_121(output)
        output = self.drop_121(output)
        output = self.bn_121(output)
        output = self.relu_121(output)
        output = self.cov3d_122(output)
        output = self.drop_122(output)
        output = self.bn_122(output)
        output12 = self.relu_122(output)

        output = self.cov3d_13(output12)

        return output
