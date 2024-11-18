import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from mmdet.models.builder import BACKBONES
from mmcv.runner import BaseModule
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule


def conv_block(in_channels, filters, kernel_size, strides, padding, mode='cba'):
    conv = nn.Conv2d(in_channels, filters, kernel_size, strides, padding, bias=False)
    bn = nn.BatchNorm2d(filters)
    act = nn.LeakyReLU(0.2)

    if mode == 'cba':
        return nn.Sequential(conv, bn, act)
    elif mode == 'cb':
        return nn.Sequential(conv, bn)
    elif mode == 'cab':
        return nn.Sequential(conv, act, bn)
    elif mode == 'ca':
        return nn.Sequential(conv, act)
    elif mode == 'c':
        return conv


class Res_block(nn.Module):
    def __init__(self, channels=64):
        super(Res_block, self).__init__()
        # first with bn
        self.conv1 = conv_block(channels, channels, 3, 1, 1, 'ca')
        # second without activation
        self.conv2 = conv_block(channels, channels, 3, 1, 1, 'c')

    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.conv2(out)
        out += x
        return out


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=256, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class CARAFE(nn.Module):
    #CARAFE: Content-Aware ReAssembly of FEatures       https://arxiv.org/pdf/1905.02188.pdf
    def __init__(self, c1, c2, kernel_size=3, up_factor=2):
        super(CARAFE, self).__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(c1, c1 // 4, 1)
        self.encoder = nn.Conv2d(c1 // 4, self.up_factor ** 2 * self.kernel_size ** 2,
                                 self.kernel_size, 1, self.kernel_size // 2)
        self.out = nn.Conv2d(c1, c2, 1)

    def forward(self, x):
        N, C, H, W = x.size()
        # N,C,H,W -> N,C,delta*H,delta*W
        # kernel prediction module
        kernel_tensor = self.down(x)  # (N, Cm, H, W)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        kernel_tensor = F.pixel_shuffle(kernel_tensor, self.up_factor)  # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor) # (N, Kup^2, H, W*S, S)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor) # (N, Kup^2, H, W, S, S)
        kernel_tensor = kernel_tensor.reshape(N, self.kernel_size ** 2, H, W, self.up_factor ** 2) # (N, Kup^2, H, W, S^2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)

        # content-aware reassembly module
        # tensor.unfold: dim, size, step
        x = F.pad(x, pad=(self.kernel_size // 2, self.kernel_size // 2,
                                          self.kernel_size // 2, self.kernel_size // 2),
                          mode='constant', value=0) # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        x = x.unfold(2, self.kernel_size, step=1) # (N, C, H, W+Kup//2+Kup//2, Kup)
        x = x.unfold(3, self.kernel_size, step=1) # (N, C, H, W, Kup, Kup)
        x = x.reshape(N, C, H, W, -1) # (N, C, H, W, Kup^2)
        x = x.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)

        out_tensor = torch.matmul(x, kernel_tensor)  # (N, H, W, C, S^2)
        out_tensor = out_tensor.reshape(N, H, W, -1)
        out_tensor = out_tensor.permute(0, 3, 1, 2)
        out_tensor = F.pixel_shuffle(out_tensor, self.up_factor)
        out_tensor = self.out(out_tensor)
        #print("up shape:",out_tensor.shape)
        return out_tensor


@BACKBONES.register_module()
class ARRD_Single_Level(BaseModule):
    def __init__(self,
                 in_channel=64,
                 mid_channel=48,
                 # out_channel=512,
                 pre_up_rate=2,
                 mode='bilinear',
                 ps_up=True,
                 # scale_factor = s,
                 init_cfg=None):
        super(ARRD_Single_Level, self).__init__(init_cfg)
        '''
        mode (str) – algorithm used for upsampling: 'nearest' | 'linear' | 'bilinear' 
        | 'bicubic' | 'trilinear' | 'area'. Default: 'bilinear'
        '''
        self.mode = mode
        # first conv without bn
        self.pre_up_rate = pre_up_rate
        self.conv_init = conv_block(in_channel, mid_channel, 3, 1, 1, 'cba')
        if self.pre_up_rate != 1:
            self.pre_up_conv = nn.ConvTranspose2d(mid_channel, mid_channel, 3, stride=pre_up_rate, padding=1,
                                                  output_padding=1)
        # self.conv_1 = conv_block(3, 16, 7, 1, 3, 'ca')
        self.res_block = Res_block(mid_channel)

        # self.dropout = nn.Dropout2d(0.1)
        self.ps_up = ps_up
        if self.ps_up:
            self.conv_final = conv_block(mid_channel, 48, 3, 1, 1, 'ca')
            self.ps = nn.PixelShuffle(4)
        else:
            self.conv_final = nn.Sequential(conv_block(mid_channel, 48, 3, 1, 1, 'ca'),
                                            conv_block(48, 3, 3, 1, 1, 'c'))

    def forward(self, x, s):
        # print(x.shape)
        x = self.conv_init(x)  # only extract one level features

        if self.pre_up_rate != 1:
            x = self.pre_up_conv(x)

        res = F.interpolate(x, scale_factor=s, mode=self.mode, align_corners=False)  # up-scale
        x = self.res_block(x)
        x = F.interpolate(x, scale_factor=s, mode=self.mode, align_corners=False)  # down-scale

        x += res  # short cut 1
        x_final = self.conv_final(x)
        if self.ps_up:
            x_final = self.ps(x_final)

        return x_final


# For FPN-like structure
@BACKBONES.register_module()
class ARRD_Multi_Level(BaseModule):
    def __init__(self,
                 in_channel=256,
                 mid_channel=128,
                 # embedding_dim=1024,
                 mode='bilinear',
                 input_number=4,  # Number of level features use for reconstruction
                 init_cfg=dict(type='Kaiming',
                               layer='Conv2d',
                               a=math.sqrt(5),
                               distribution='uniform',
                               mode='fan_in',
                               nonlinearity='leaky_relu')):
        super(ARRD_Multi_Level, self).__init__(init_cfg)
        '''
        mode (str) – algorithm used for upsampling: 'nearest' | 'linear' | 'bilinear' 
        | 'bicubic' | 'trilinear' | 'area'. Default: 'bilinear'
        '''
        self.mode = mode
        self.input_number = input_number
        ## Level of features to use
        if self.input_number >= 4:
            # self.linear_c4 = MLP(input_dim=in_channel, embed_dim=embedding_dim)
            self.linear_c4 = MLP(input_dim=in_channel, embed_dim=in_channel * 4)
        if self.input_number >= 3:
            self.linear_c3 = MLP(input_dim=in_channel, embed_dim=in_channel * 4)
        self.linear_c2 = MLP(input_dim=in_channel, embed_dim=in_channel * 4)
        self.linear_c1 = MLP(input_dim=in_channel, embed_dim=in_channel * 4)

        self.linear_fuse = ConvModule(
            in_channels=in_channel * input_number,
            out_channels=mid_channel,
            kernel_size=1,
            # norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        self.res_block = Res_block(mid_channel)
        self.conv_final = conv_block(mid_channel,3, 3, 1, 1, 'c')
        # self.dropout = nn.Dropout2d(0.1)
        self.c1 = CARAFE(in_channel, in_channel, up_factor=8)
        self.c2 = CARAFE(in_channel, in_channel, up_factor=4)
        self.c3 = CARAFE(in_channel, in_channel, up_factor=2)
        self.c4 = CARAFE(mid_channel, mid_channel, up_factor=2)
        self.c5 = CARAFE(mid_channel, mid_channel, up_factor=2)


    def forward(self, x):
        c1, c2, c3, c4, _ = x
        # print(c1.shape, c4.shape)
        n = c4.shape[0]
        if self.input_number >= 4:
            _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2] * 2, c4.shape[3] * 2)
            _c4 = self.c1(_c4)
            # _c4 = F.interpolate(_c4, size=(c1.shape[2] * 2, c1.shape[3] * 2), mode='bilinear', align_corners=False)

        if self.input_number >= 3:
            _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2] * 2, c3.shape[3] * 2)
            _c3 = self.c2(_c3)
            # _c3 = F.interpolate(_c3, size=(c1.shape[2] * 2, c1.shape[3] * 2), mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2] * 2, c2.shape[3] * 2)
        _c2 = self.c3(_c2)
        # _c2 = F.interpolate(_c2, size=(c1.shape[2] * 2, c1.shape[3] * 2), mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2] * 2, c1.shape[3] * 2)

        # Linear Fusion
        if self.input_number == 4:
            x = self.linear_fuse(torch.cat([_c1, _c2, _c3, _c4], dim=1))
        elif self.input_number == 3:
            x = self.linear_fuse(torch.cat([_c1, _c2, _c3], dim=1))
        else:
            x = self.linear_fuse(torch.cat([_c1, _c2], dim=1))

        # ARRD decoder
        res = self.c4(x)#(1, 128, 64, 64)
        x = self.res_block(x)
        x = self.c5(x)
        x += res

        x_final = self.conv_final(x)
        # x_final = self.ps(x_final)

        return x_final


# For FPN-like structure
@BACKBONES.register_module()
class ARRD_Multi_Level_type2(BaseModule):
    def __init__(self,
                 in_channel=[512, 256, 128],
                 mid_channel=128,
                 # embedding_dim=1024,
                 mode='bilinear',
                 input_number=3,  # Number of level features use for reconstruction
                 init_cfg=dict(type='Kaiming',
                               layer='Conv2d',
                               a=math.sqrt(5),
                               distribution='uniform',
                               mode='fan_in',
                               nonlinearity='leaky_relu')):
        super(ARRD_Multi_Level_type2, self).__init__(init_cfg)
        '''
        mode (str) – algorithm used for upsampling: 'nearest' | 'linear' | 'bilinear' 
        | 'bicubic' | 'trilinear' | 'area'. Default: 'bilinear'
        '''
        self.mode = mode
        self.input_number = input_number
        ## Level of features to use
        if self.input_number >= 3:
            self.linear_c1 = MLP(input_dim=in_channel[0], embed_dim=in_channel[0] * 4)
        self.linear_c2 = MLP(input_dim=in_channel[1], embed_dim=in_channel[1] * 4)
        self.linear_c3 = MLP(input_dim=in_channel[2], embed_dim=in_channel[2] * 4)

        self.linear_fuse = ConvModule(
            in_channels=in_channel[0] + in_channel[1] + in_channel[2],
            out_channels=mid_channel,
            kernel_size=1,
            # norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        self.res_block = Res_block(mid_channel)
        self.conv_final = conv_block(mid_channel,3, 3, 1, 1, 'c')
        # self.dropout = nn.Dropout2d(0.1)
        self.c1 = CARAFE(in_channel[0], in_channel[0], up_factor=8)
        self.c2 = CARAFE(in_channel[1], in_channel[1], up_factor=4)
        self.c3 = CARAFE(in_channel[2], in_channel[2], up_factor=2)
        self.c4 = CARAFE(mid_channel, mid_channel, up_factor=2)


    def forward(self, x, s):
        c1, c2, c3 = x
        # print(c1.shape, c4.shape)
        n = c1.shape[0]

        if self.input_number >= 3:
            _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2] * 2, c3.shape[3] * 2)
            _c3 = self.c3(_c3)
            # _c3 = F.interpolate(_c3, size=(c1.shape[2] * 2, c1.shape[3] * 2), mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2] * 2, c2.shape[3] * 2)
        _c2 = self.c2(_c2)
        # _c2 = F.interpolate(_c2, size=(c1.shape[2] * 2, c1.shape[3] * 2), mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2] * 2, c1.shape[3] * 2)
        _c1 = self.c1(_c1)

        # Linear Fusion
        if self.input_number == 3:
            x = self.linear_fuse(torch.cat([_c1, _c2, _c3], dim=1))
        else:
            x = self.linear_fuse(torch.cat([_c1, _c2], dim=1))

        # ARRD decoder
        res = self.c4(x)#(1, 128, 64, 64)
        x = self.res_block(x)
        x = self.c4(x)
        x += res

        x_final = self.conv_final(x)
        # x_final = self.ps(x_final)

        return x_final


if __name__ == "__main__":
    SR_decoder = ARRD_Multi_Level()
    x = [torch.ones([1, 256, 128, 128]),
         torch.ones([1, 256, 64, 64]),
         torch.ones([1, 256, 32, 32]),
         torch.ones([1, 256, 16, 16]),
         torch.ones([1, 256, 8, 8])]
    # x = [
    #     torch.ones([1, 512, 20, 20]),
    #     torch.ones([1, 256, 40, 40]),
    #     torch.ones([1, 128, 80, 80]),
    # ]

    mlp = MLP()
    x_out = SR_decoder(x)
    # x_out = SR_decoder(x, s=1.6)
    print(x_out.shape)
    # print()
