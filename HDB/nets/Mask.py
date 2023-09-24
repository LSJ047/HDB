import torch
import torch.nn as nn


class MConv7x7(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MConv7x7, self).__init__()
        # 定义一个7x7的卷积层
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3)
        # 定义激活函数（你也可以根据需要选择其他的激活函数）
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 将输入通过卷积层和激活函数
        x = self.conv(x)
        # x = self.relu(x)
        return x


# # 使用示例
# mconv_layer = MConv7x7(in_channels=1, out_channels=64)  # 假设输入通道数为1，输出通道数为64
#
# # 使用 mconv_layer 进行前向传播
# input_data = torch.randn(1, 1, 192, 256)  # 假设输入数据的形状为(1, 1, 192, 256)
# output = mconv_layer(input_data)
#
# # 输出的形状为 (1, 64, 192, 256)
# print(output.shape)
