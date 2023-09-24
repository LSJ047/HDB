import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    """下采样"""

    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class DecoderBlock(nn.Module):
    """上采样"""

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x1 = self.up_conv(x1)
        x = torch.cat([x2, x1], dim=1)  # Concatenate skip connection
        x = self.relu(x)
        x = self.conv(x)
        return x


class Structural_Context_Extractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Structural_Context_Extractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, stride=1)
        self.encoder1 = EncoderBlock(64, 128)
        self.encoder2 = EncoderBlock(128, 256)

        self.decoder2 = DecoderBlock(256, 128)
        self.decoder1 = DecoderBlock(128, 64)
        self.conv2 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)

        x = self.decoder2(x3, x2)
        x = self.decoder1(x, x1)
        x = self.conv2(x)

        return x

# # 使用示例
# s_layer = Structural_Context_Extractor(in_channels=1, out_channels=64)  # 假设输入通道数为1，输出通道数为64
#
# # 使用 mconv_layer 进行前向传播
# input_data = torch.randn(1, 256, 192)  # 假设输入数据的形状为(1, 192, 256)
# output = s_layer(input_data)
#
# # 输出的形状为 (1, 64, 192, 256)
# print(output.shape)