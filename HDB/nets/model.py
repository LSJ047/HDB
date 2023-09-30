import torch
import torch.nn as nn

from nets.Structural_Context_Extractor import Structural_Context_Extractor
from nets.Parameter_Predictor import Parameter_Predictor
from nets.Mask import MConv7x7
from utils.loss import Distribution_for_entropy2
from utils.backround import Low_bound


class HDB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HDB, self).__init__()
        self.S = Structural_Context_Extractor(in_channels, out_channels)
        self.M = MConv7x7(in_channels, out_channels)
        self.P = Parameter_Predictor(out_channels, out_channels, 3)
        self.gaussin_entropy_func = Distribution_for_entropy2(distribution_type= 'logistic')

    def forward(self, msb, lsb):
        lable =lsb
        cs = self.S(msb)
        cl = self.M(lsb)
        p = self.P(cl, cs)

        prob = self.gaussin_entropy_func(lable, p)
        prob = Low_bound.apply(prob)  # Clamp,最小值为1e-6

        bits = -torch.sum(torch.log2(prob), dim=[1, 2, 3])  # 似然损失转化成bit
        return bits

# 假设输入通道数为 1，输出通道数为 64
# model = HDB(in_channels=1, out_channels=64)
#
# # 随机生成输入数据（假设输入数据形状为 (batch_size, channels, height, width)）
# msb = torch.randn(1, 1, 192, 256)
# lsb = torch.randn(1, 1, 192, 256)
#
# # 进行前向传播
# output_params = model(msb, lsb)
#
#
# # 打印输出的参数形状
# print(f"p shape: {output_params.shape}")

