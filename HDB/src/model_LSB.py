#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/13
# @Author  : Tiantian Li
# @Github  : https://github.com/Sweethyh
# @Software: PyCharm
# @File    : model_LSB.py

from collections import defaultdict
from typing import (DefaultDict, Generator, KeysView, List, NamedTuple,
                    Optional, Tuple, Any, Union)

import numpy as np
import torch
import copy
from torch import nn

import torch.nn.functional as F
from src.util import Base,checkboard,Conv2dReLU,DenseBlock,DenseLayer,Identity,ResidualBlock,space_to_depth,depth_to_space,\
    MaskedConv2d,MaskResBlock,img2patch,img2patch_padding,pad_img
from src.enc_dec import encode_torchac_cdf
from utils.loss import Distribution_for_entropy2
from utils.backround import Low_bound

class SCG(nn.Module):
    def __init__(self, n_feats) -> None:
        super().__init__()
        # assert configs.scale >= 0, configs.scale
        # rgb_scale = (inchannel == 3)

        self.conv1 = nn.Conv2d(1, n_feats, stride=1, kernel_size=3,padding=1)
        self.down = nn.Conv2d(n_feats, n_feats*2, stride=2, kernel_size=3,padding=1)
        self.conv2 = nn.Sequential(
                    nn.Conv2d(n_feats*2, n_feats*4, stride=2, kernel_size=3,padding=1),
                    nn.ConvTranspose2d(n_feats*4, n_feats*2, stride=2, kernel_size=3, padding=1,output_padding=0)
        )
        self.conv3 = nn.Sequential(
                    nn.Conv2d(n_feats*4, n_feats*4, stride=1, kernel_size=1),
                    nn.GELU(),
                    nn.ConvTranspose2d(n_feats*4, n_feats*2, stride=2, kernel_size=3, padding=1,output_padding=1)#
        )
        self.conv4 = nn.Sequential(
                    nn.Conv2d(n_feats*3, n_feats, stride=1, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(n_feats, n_feats, stride=1, kernel_size=3,padding=1)
        )

    def forward(self, DC_MSB):
        x1= self.conv1(DC_MSB)
        x2=self.down(x1)
        x3=self.conv2(x2)
        x=self.conv3(torch.cat([x2,x3],dim=1))
        x=self.conv4(torch.cat([x1,x],dim=1))
        return x


class ParameterPredictor(nn.Module):
    def __init__(self, n_feats, n_mixtures) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
                    nn.Conv2d(n_feats, n_feats, stride=1, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(n_feats, n_feats, stride=1, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(n_feats, n_feats, stride=1, kernel_size=1),
                    nn.GELU(),
        )
        self.conv2 = nn.Sequential(
                    nn.Conv2d(n_feats, n_feats, stride=1, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(n_feats, 3*n_mixtures, stride=1, kernel_size=1)
        )


    def forward(self, C_l,C_s):
        x1= torch.cat([C_l,C_s],dim=1)
        x2=self.conv1(x1)
        x3=self.conv2(x1+x2)

        return x3


# 压缩网络
class Compressor(nn.Module):
    def __init__(self, args, inchannel=1, writer=None) -> None:
        super().__init__()
        # assert configs.scale >= 0, configs.scale
        # rgb_scale = (inchannel == 3)

        self.args = args

        self.group_num = args.group_num #

        self.writer = writer
        _, height, width = args.input_size


        self.padding_constant = torch.nn.ConstantPad2d(3, 0)
        self.conv_pre = MaskedConv2d('A', in_channels=inchannel, out_channels=self.args.n_feats, kernel_size=7, stride=1, padding=0)
        self.SCG=SCG(self.args.n_feats)
        self.params=ParameterPredictor(128,self.args.n_mixtures) # cl + cs

        self.gaussin_entropy_func = Distribution_for_entropy2(distribution_type=self.args.distribution_type)




    def forward(self,  # type: ignore
               DC_MSB,DC_LSB,mask_alpha =None
                ) -> List[Any]:
       lable =DC_LSB
       DC_LSB=self.padding_constant(DC_LSB)
       C_l=self.conv_pre(DC_LSB) # mask conv
       C_s=self.SCG(DC_MSB)
       params=self.params(C_l,C_s)
       prob = self.gaussin_entropy_func(lable, params,self.args) # 看作一个损失函数  计算label（真实的量化值）和估计的参数之间的似然  72 1 8 8
       prob = Low_bound.apply(prob) # Clamp,最小值为1e-6

       bits = -torch.sum(torch.log2(prob),dim=[1,2,3]) # 似然损失转化成bit
       return bits

    def inference(self, x, C_s,range_low_bond, range_high_bond):


        label = torch.arange(start=range_low_bond, end=range_high_bond + 1, dtype=torch.float,
                             device=x.device)
        label = label.unsqueeze(0).unsqueeze(0).unsqueeze(0)/255.
        # label = x
        # x = x / 255. # 3 1 13 13
        # label has the size of [1,1,1,range]
        # x=self.padding_constant(x)
        C_l=self.conv_pre(x)
        # C_s=self.SCG(DC_MSB)
        # C_s=C_s.mean(dim=(2,3),keepdim=True) # 2023/9/9
        params=self.params(C_l,C_s)
        params = params.repeat(1, 1, 1, int(range_high_bond - range_low_bond + 1))

        prob = self.gaussin_entropy_func(label, params,self.args)

        return prob

    def compress(self,  # type: ignore
               DC_MSB,DC_LSB,fout,mask_alpha =None
                ) -> List[Any]:


         # 编码
        C_s=self.SCG(pad_img(DC_MSB.squeeze(0))[0])
        # C_s=self.SCG(DC_MSB)  # Xl 在该点的结构信息


        # min max value
        code_block_size=self.args.code_size
        yuv_low_bound=[] # 3个通道中的最小值
        yuv_high_bound =[] # 3个通道中的最大值
        for a,item in enumerate(DC_LSB):
            yuv_low_bound.append(np.int16(torch.min(item).cpu().item())) # 3个通道中的最小值
            yuv_high_bound.append(np.int16(torch.max(item).cpu().item()))
            fout.write(yuv_low_bound[a])
            fout.write(yuv_high_bound[a])

        # compress first channel=======================================

        # alpha=mask_alpha[0] if mask_alpha is not None else None
        padding_z = [(int(np.ceil(tmp / code_block_size)) * code_block_size - tmp) for tmp in DC_LSB.shape[2:]]
        paddings = (0, padding_z[1], 0, padding_z[0])
        enc_z=F.pad(DC_LSB,paddings,'replicate')

        padding_z = [(int(np.ceil(tmp / code_block_size)) * code_block_size - tmp) for tmp in C_s.shape[2:]]
        paddings = (0, padding_z[1], 0, padding_z[0])
        enc_msb=F.pad(C_s,paddings,'replicate')

        enc_z= img2patch(enc_z, code_block_size, code_block_size, code_block_size)
        enc_msb= img2patch(enc_msb, code_block_size, code_block_size, code_block_size)

        paddings = (3,3, 3, 3)
        enc_z = F.pad(enc_z, paddings, "constant")
        enc_msb=F.pad(enc_msb,paddings,"constant")
        # enc_msb= img2patch_padding(enc_msb, code_block_size+6, code_block_size+6, code_block_size,3)
        # enc_alpha = F.pad(enc_alpha, paddings, "constant" ) if alpha is not None else None
        bytes_sum=0
        bytes_per_slice=[]
        for h_i in range(code_block_size):
            for w_i in range(code_block_size):
                cur_ct = copy.deepcopy(enc_z[:, :, h_i:h_i + 7, w_i:w_i + 7]) # 7x7的窗口
                cur_ct[:, :, 7 // 2 + 1:7, :] = 0.
                cur_ct[:, :, 7 // 2, 7 // 2:7] = 0. # mask掉不可见的信息（未解码） (3,3)正中间的地方也看不见
                cur_MSB=enc_msb[:, :, h_i:h_i + 7, w_i:w_i + 7]#enc_msb[:, :, h_i:h_i + 7, w_i:w_i + 7]
                C_s_point=cur_MSB[:,:, 7 // 2:7 // 2+1, 7 // 2:7 // 2+1]
                prob=self.inference(cur_ct,C_s_point,yuv_low_bound[0] ,yuv_high_bound[0])#,alpha=cur_alpha )

                j_b =(enc_z[:, 0, h_i + 3, w_i + 3]-yuv_low_bound[0]).long()
                bytes=encode_torchac_cdf(prob[None,...].cpu(), j_b[None,...],  True, fout)
                bytes_sum+=bytes
        bytes_per_slice.append(bytes_sum)


        return bytes_per_slice
