import torch
import torch.nn as nn


class Parameter_Predictor(nn.Module):
        def __init__(self, in_channels_cl, in_channels_cs, num_parameters):
            super(Parameter_Predictor, self).__init__()

            # 第一个分支，将 Cl 与 Cs 进行拼接
            self.concat = nn.Conv2d(in_channels_cl + in_channels_cs, in_channels_cl,
                                    kernel_size=1, stride=1, padding=0)

            # 第二个分支，对拼接后的特征图进行 3 个卷积操作
            self.conv1 = nn.Conv2d(in_channels_cl, in_channels_cs,
                                   kernel_size=1, stride=1, padding=0)
            self.conv2 = nn.Conv2d(in_channels_cl, in_channels_cs,
                                   kernel_size=1, stride=1, padding=0)
            self.conv3 = nn.Conv2d(in_channels_cl, in_channels_cs,
                                   kernel_size=1, stride=1, padding=0)

            # 第三个分支，将第一个分支和第二个分支的结果进行跳跃连接
            self.skip_connect = nn.Conv2d(in_channels_cl + in_channels_cs, in_channels_cl,
                                          kernel_size=1, stride=1, padding=0)

            # 第四个分支，对跳跃连接后的特征图进行 2 个卷积操作
            self.conv4 = nn.Conv2d(in_channels_cl, in_channels_cs,
                                   kernel_size=1, stride=1, padding=0)
            self.conv5 = nn.Conv2d(in_channels_cl, num_parameters, kernel_size=1, stride=1,
                                   padding=0)

        def forward(self, x_cl, x_cs):
            # 第一个分支，将 Cl 与 Cs 进行拼接
            x1 = torch.cat([x_cl, x_cs], dim=1)
            x1 = self.concat(x1)
            # 第二个分支，对拼接后的特征图进行 3 个卷积操作
            x2 = self.conv1(x1)
            x2 = self.conv2(x2)
            x2 = self.conv3(x2)

            # 第三个分支，将第一个分支和第二个分支的结果进行跳跃连接
            x3 = torch.cat([x1, x2], dim=1)
            x3 = self.skip_connect(x3)

            # 第四个分支，对跳跃连接后的特征图进行 2 个卷积操作
            x4 = self.conv4(x3)
            x4 = self.conv5(x4)

            return x4


# 使用示例
model = Parameter_Predictor(in_channels_cl=64, in_channels_cs=64, num_parameters=3)
# 假设输入通道数分别为3和1，输出参数数为3

# 生成随机输入数据
input_data_cl = torch.randn(1, 64, 192, 256)
input_data_cs = torch.randn(1, 64, 192, 256)

# 使用模型进行前向传播
# sigma, pi, mu = model(input_data_cl, input_data_cs)
# outputs = model(input_data_cl, input_data_cs)

# # 输出的形状
# print(sigma.shape)
# print(pi.shape)
# print(mu.shape)
# print(outputs[0][0].shape)
