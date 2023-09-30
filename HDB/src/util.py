import torch
import torch.nn as nn
import scipy.fft as fft
from collections import Counter
from PIL  import Image
# import cv2
import numpy as np
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

def conv(in_channels: int,
         out_channels: int,
         kernel_size: int,
         bias: bool = True,
         rate: int = 1,
         stride: int = 1) -> nn.Conv2d:
    padding = kernel_size // 2 if rate == 1 else rate
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, stride=stride, dilation=rate,
        padding=padding, bias=bias)


def tensor_round(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x - 0.001)


class Base(torch.nn.Module):
    """
    The base class for modules. That contains a disable round mode
    """

    def __init__(self):
        super().__init__()

    def _set_child_attribute(self, attr, value):
        r"""Sets the module in rounding mode.

        This has any effect only on certain modules if variable type is
        discrete.

        Returns:
            Module: self
        """
        if hasattr(self, attr):
            setattr(self, attr, value)

        for module in self.modules():
            if hasattr(module, attr):
                setattr(module, attr, value)
        return self

    def set_temperature(self, value):
        self._set_child_attribute("temperature", value)

    def enable_hard_round(self, mode=True):
        self._set_child_attribute("hard_round", mode)

    def disable_hard_round(self, mode=True):
        self.enable_hard_round(not mode)


def checkboard(imgs,way='ver'):
    # img=cv2.imread(path, 0)
    b,c,h,w=imgs.shape
    padd_image=F.pad(imgs,(0,(w+1)//2*2-w,0,(h+1)//2*2-h),mode='constant',value=0)

    anchor_h=(h+1)//2
    anchor_w=(w+1)//2
    _,_,padd_h,padd_w=padd_image.shape
    # plotHistogram([img])
    y_anchor_encode = torch.zeros((b,c,(h+1)//2, padd_w)).to(imgs.device) # h变为原来的一半
    y_non_anchor_encode = torch.zeros((b,c,padd_h-(h+1)//2, padd_w)).to(imgs.device)

    y_anchor_encode[...,:, 0::2] =padd_image[...,0::2, 0::2] # 偶数行 列
    y_anchor_encode[...,:,1::2] = padd_image[...,1::2, 1::2] # 奇数行 列


    # y_non_anchor_encode = torch.zeros((b,c,padd_h-(h+1)//2, padd_w-(w+1)//2)).to(imgs.device) # h变为原来的一半

    y_non_anchor_encode[..., :,0::2] = padd_image[..., 0::2, 1::2]
    y_non_anchor_encode[...,:, 1::2] = padd_image[ ...,1::2, 0::2]

    if way=='hor':
        y_anchor_encode = torch.zeros((b,c,padd_h, (w+1)//2)).to(imgs.device) # h变为原来的一半
        y_non_anchor_encode = torch.zeros((b,c,padd_h, (w+1)//2)).to(imgs.device)
        # padd_image=F.pad(imgs,(0,(w+1)//2*2-w,0,0),mode='constant',value=0)
        # y_anchor_encode = torch.zeros((b,c,h, (w+1)//2)).to(imgs.device) # w变为原来的一半 # W变为原来的一半
        y_anchor_encode[..., 0::2, :] =padd_image[...,0::2, 0::2] # 偶数行 列
        y_anchor_encode[...,1::2, :] = padd_image[...,1::2, 1::2] # 奇数行 列

        # y_non_anchor_encode = torch.zeros((b,c,h, w//2)).to(imgs.device) # h变为原来的一半 # W变为原来的一半

        y_non_anchor_encode[..., 0::2, :] =padd_image[..., 0::2, 1::2]
        y_non_anchor_encode[..., 1::2, :] =padd_image[ ...,1::2, 0::2]

        y_anchor_encode=y_anchor_encode[...,:h,:anchor_w]
        y_non_anchor_encode=y_non_anchor_encode[...,:h,:w-anchor_w]
    else:
        y_anchor_encode=y_anchor_encode[...,:anchor_h,:w]
        y_non_anchor_encode=y_non_anchor_encode[...,:h-anchor_h, :w]
    return y_anchor_encode,y_non_anchor_encode




class Conv2dReLU(Base):
    def __init__(
            self, n_inputs, n_outputs, kernel_size=3, stride=1, padding=0,
            bias=True):
        super().__init__()

        self.nn = nn.Conv2d(n_inputs, n_outputs, kernel_size, padding=padding)

    def forward(self, x):
        h = self.nn(x)

        y = F.relu(h)


        return y


class ResidualBlock(Base):
    def __init__(self, n_channels, kernel, Conv2dAct):
        super().__init__()

        self.nn = torch.nn.Sequential(
            Conv2dAct(n_channels, n_channels, kernel, padding=1),
            torch.nn.Conv2d(n_channels, n_channels, kernel, padding=1),
            )

    def forward(self, x):
        h = self.nn(x)
        h = F.relu(h + x)
        return h


class DenseLayer(Base):
    def __init__(self, args, n_inputs, growth, Conv2dAct):
        super().__init__()

        conv1x1 = Conv2dAct(
                n_inputs, n_inputs, kernel_size=1, stride=1,
                padding=0, bias=True)

        self.nn = torch.nn.Sequential(
            conv1x1,
            Conv2dAct(
                n_inputs, growth, kernel_size=3, stride=1,
                padding=1, bias=True),
            )

    def forward(self, x):
        h = self.nn(x)

        h = torch.cat([x, h], dim=1)
        return h


class DenseBlock(Base):
    def __init__(
            self, args, n_inputs, n_outputs, kernel, Conv2dAct):
        super().__init__()
        depth = args.densenet_depth

        future_growth = n_outputs - n_inputs

        layers = []

        for d in range(depth):
            growth = future_growth // (depth - d)

            layers.append(DenseLayer(args, n_inputs, growth, Conv2dAct))
            n_inputs += growth
            future_growth -= growth

        self.nn = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)


class Identity(Base):
    def __init__(self):
        super.__init__()

    def forward(self, x):
        return x

class CheckboardMaskedConv2d(nn.Conv2d):
    """
    if kernel_size == (5, 5)
    then mask:
        [[0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.]]
    0: non-anchor
    1: anchor
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #
        # self.register_buffer("mask", torch.zeros_like(self.weight.data))
        #
        # self.mask[:, :, 0::2, 1::2] = 1
        # self.mask[:, :, 1::2, 0::2] = 1
        self.register_buffer("mask", torch.ones_like(self.weight.data))

        self.mask[:, :, 0::2, 1::2] = 0
        self.mask[:, :, 1::2, 0::2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        out = super().forward(x)

        return out

def space_to_depth(x):
    xs = x.size()
    # Pick off every second element
    x = x.view(xs[0], xs[1], xs[2] // 2, 2, xs[3] // 2, 2)
    # Transpose picked elements next to channels.
    x = x.permute((0, 1, 3, 5, 2, 4)).contiguous()
    # Combine with channels.
    x = x.view(xs[0], xs[1] * 4, xs[2] // 2, xs[3] // 2)
    return x


def depth_to_space(x):
    xs = x.size()
    # Pick off elements from channels
    x = x.view(xs[0], xs[1] // 4, 2, 2, xs[2], xs[3])
    # Transpose picked elements next to HW dimensions.
    x = x.permute((0, 1, 4, 2, 5, 3)).contiguous()
    # Combine with HW dimensions.
    x = x.view(xs[0], xs[1] // 4, xs[2] * 2, xs[3] * 2)
    return x


def split(z,split_mode='checkboard',way='ver'):
        '''
            棋盘格划分还是对半划分？
        '''
        B, C, H, W = z.size()
        # X-ray图像对半分
        z1 = z[:,:,:, :W//2] # B C H W/2
        z2 = z[:,:,:, W//2:]
        # z2_mirror=torch.flip(z2,[3])
        z2=torch.flip(z2,[3])

        if split_mode=='checkboard':
            z1, z2=checkboard(z,way=way) # 垂直减半


        return z1, z2

def z_order(block):
    H,W=block.shape
    arr=np.zeros((64,),dtype=block.dtype)
    ZigZag = np.array([
        0, 1, 5, 6, 14, 15, 27, 28,
        2, 4, 7, 13, 16, 26, 29, 42,
        3, 8, 12, 17, 25, 30, 41, 43,
        9, 11, 18, 24, 31, 40, 44, 53,
        10, 19, 23, 32, 39, 45, 52, 54,
        20, 22, 33, 38, 46, 51, 55, 60,
        21, 34, 37, 47, 50, 56, 59, 61,
        35, 36, 48, 49, 57, 58, 62, 63])
    for k in range(64):
        tmp = block[int(k / 8),k % 8]
        arr[ZigZag[k]] = tmp
    return arr

def block_dct_float_lossless(image,b):
    # image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if len(image.shape)!=2:
        print(image.shape)
    h,w=image.shape
    # b=b*2
    image=image.astype(np.float32)
    padded_h = (h + b - 1) // b * b
    padded_w = (w + b - 1) // b * b
    pad_h = padded_h - h
    pad_w = padded_w - w
    # pad way1  entropy is smaller
    padd_img = np.pad(image, pad_width=((0, padded_h - h), (0, padded_w - w)), mode='edge')
    # padd way 2
    # padd_img = np.pad(image, pad_width=((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
    #                   mode='edge')


    dct_image=np.zeros_like(padd_img, dtype=np.float32)
    # dct_round_diff = np.zeros_like(padd_img, dtype=np.float32)
    # idct_image=np.zeros_like(padd_img, dtype=np.float32)

    # arrange = np.zeros((64,padded_h // b, padded_w // b))

    # 将图像划分为 8x8 块
    for i in range(0, padd_img.shape[0], b):
        for j in range(0, padd_img.shape[1], b):
            block = padd_img[..., i:i+b, j:j+b]
            dct_block = np.round(fft.dct(fft.dct(block, axis=0, norm='ortho'), axis=1, norm='ortho'))  # 取整
            # idct_block=Fidct2D_II((dct_block))
            # # 反变换
            # idct_block=np.int16(np.trunc(np.clip(fft.idct(fft.idct(dct_block, axis=0, norm='ortho'), axis=1, norm='ortho')+0.5,0,255)))
            #
            # idct_image[..., i:i + b, j:j + b] = idct_block
            dct_image[..., i:i + b, j:j + b] = dct_block
            # # dct_round_diff[i:i + b, j:j + b] = block-idct_block

            # zigzag
            # z_order_result = z_order(dct_block)
            # arrange[:,i // b, j // b] = z_order_result
    return dct_image
    # return arrange # B C*64 H/8 W/8
def block_dct_DC(image,b,start,num):
    '''
        zigzag
    '''
    # image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    h,w=image.shape
    image=image.astype(np.float32)
    padded_h=(h+b-1)//b*b
    padded_w=(w+b-1)//b*b
    padd_img = np.pad(image, pad_width=((0, padded_h - h), (0,  padded_w - w)), mode='edge')
    arrange=np.zeros((1,1,padded_h//b,padded_w//b),dtype=np.float32)
    # 将图像划分为 8x8 块

    for i in range(0, padd_img.shape[0], b):
        for j in range(0, padd_img.shape[1], b):
            block = padd_img[i:i+b, j:j+b]
            arrange[:,0,i//b,j//b]=block[0,0]
    return arrange
def block_dct_coe_filter(image,b,start,num):
    '''
        zigzag
    '''
    # image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    h,w=image.shape
    image=image.astype(np.float32)
    padded_h=(h+b-1)//b*b
    padded_w=(w+b-1)//b*b
    padd_img = np.pad(image, pad_width=((0, padded_h - h), (0,  padded_w - w)), mode='edge')
    dct_image=np.zeros_like(padd_img, dtype=np.float32)
    # # dct_round_diff = np.zeros_like(padd_img, dtype=np.float32)
    # idct_image=np.zeros_like(padd_img, dtype=np.float32)
    arrange=np.zeros((num,1,padded_h//b,padded_w//b),dtype=np.float32)
    # 将图像划分为 8x8 块

    for i in range(0, padd_img.shape[0], b):
        for j in range(0, padd_img.shape[1], b):
            block = padd_img[i:i+b, j:j+b]

            # # zig scan
            z_order_result=z_order(block)
            arrange[:,0,i//b,j//b]=z_order_result[start:start+num]
    return arrange

class MaskedConv2d(torch.nn.Conv2d):
    def __init__(self, mask_type, alpha=None,*args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0
        if alpha is not None:
            self.mask=self.mask*alpha

    def forward(self, x,alpha=None):

        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

class MaskResBlock(torch.nn.Module):
    def __init__(self, internal_channel,alpha=None):
        super(MaskResBlock, self).__init__()

        self.conv1 = MaskedConv2d('B', alpha=alpha,in_channels=internal_channel, out_channels=internal_channel, kernel_size=3, stride=1, padding=0)
        self.conv2 = MaskedConv2d('B', alpha=alpha,in_channels=internal_channel, out_channels=internal_channel, kernel_size=3, stride=1, padding=0)
        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x[:,:,2:-2,2:-2]

# 划分为h w 的block，在通道维度上拼接
def img2patch(x, h, w, stride):
    size = x.size()
    x_tmp = x[:, :, 0:h, 0:w]
    for i in range(0, size[2], stride):
        for j in range(0, size[3], stride):
            x_tmp = torch.cat((x_tmp, x[:, :, i:i+h, j:j+w]), dim=0)
    return x_tmp[size[0]::, :, :, :]
def img2patch_padding(x, h, w, stride, padding):
    size = x.size()
    x_tmp = x[:, :, 0:h, 0:w]
    for i in range(0, size[2]-2*padding, stride):
        for j in range(0, size[3]-2*padding, stride):
            x_tmp = torch.cat((x_tmp, x[:, :, i:i+h, j:j+w]), dim=0)
    return x_tmp[size[0]::, :, :, :]


def pad_img(img):

    C, H, W = img.shape

    pad_h = (H % 2)
    pad_w = (W % 2)
    padding = (0,pad_w,0,pad_h)

    pad_rep = torch.nn.ReplicationPad2d(padding)
    img = pad_rep(img.unsqueeze(0))
    # img = img[0]

    return img, (pad_h, pad_w)
