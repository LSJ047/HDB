import torch
import torch.nn as nn

from nets.Structural_Context_Extractor import Structural_Context_Extractor
from nets.Parameter_Predictor import Parameter_Predictor
from nets.Mask import MConv7x7


class HDB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HDB, self).__init__()
        self.S = Structural_Context_Extractor(in_channels, out_channels)
        self.M = MConv7x7(in_channels, out_channels)
        self.P = Parameter_Predictor(out_channels, out_channels, 3)

    def forward(self, msb, lsb):
        cs = self.S(msb)
        cl = self.M(lsb)
        p = self.P(cl, cs)

        pi = p[:, :1]
        mu = p[:, 1:2]
        sigma = p[:, 2:]

        pmf_logical = pi*torch.sigmoid((mu - 0.5) / sigma)
        return pmf_logical

# # 假设输入通道数为 1，输出通道数为 64
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

