import torch
import random
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_

class Conv(nn.Module):
    def __init__(self, inp, oup, kernal_size, stride, padding, group=1):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(*[
            nn.Conv2d(inp, oup, kernel_size=kernal_size, stride=stride, padding=padding, groups=group),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=False)
        ])

    def forward(self, x):
        return self.conv(x)
#############################################################################################################
class Stem(nn.Module):
    def __init__(self, inp, oup):
        super(Stem, self).__init__()
        self.conv1 = Conv(inp, oup, kernal_size=3, stride=2, padding=1)
        self.conv2 = Conv(oup, oup, kernal_size=3, stride=1, padding=1, group=oup)
        self.conv3 = Conv(oup, oup, kernal_size=1, stride=1, padding=0)
        self.conv4 = Conv(oup, oup, kernal_size=3, stride=2, padding=1, group=oup)

    def forward(self, x):
        x = self.conv1(x)
        short_cut = x
        x = self.conv2(x)
        x = self.conv3(x)
        x = x+short_cut
        out = self.conv4(x)
        return out

class Res_block(nn.Module):
    def __init__(self, inp, oup):
        super(Res_block, self).__init__()
        self.conv_d = Conv(inp, oup, kernal_size=3, stride=2, padding=1)
        self.dw_conv = Conv(oup, oup, kernal_size=3, stride=1, padding=1, group=oup)
        self.pw_conv = Conv(oup, oup, kernal_size=1, stride=1, padding=0)
        self.conv_s = Conv(inp, oup, kernal_size=3, stride=2, padding=1)

    def forward(self, x):
        short_cut = self.conv_s(x)
        x = self.pw_conv(self.dw_conv(self.conv_d(x)))
        out = x+short_cut
        return out

class local_branch(nn.Module):
    def __init__(self, inp=3, oup=[16, 64, 128, 256], k=16,  mode=0):
        super(local_branch, self).__init__()
        self.mode = mode
        self.k = k
        self.stem = Stem(inp, oup[0])
        self.res1 = Res_block(oup[0], oup[1])
        self.res2 = Res_block(oup[1], oup[2])
        self.res3 = Res_block(oup[2], oup[3])

        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(oup[3], 3*k)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.stem(x)
        x = self.res3(self.res2(self.res1(x)))
        out = x
        B, C, _, _ = x.shape

        gap_w = self.gap(out).view(B, C)
        gmp_w = self.gmp(out).view(B, C)
        weights = gap_w+gmp_w
        w = self.act(self.fc(weights))
        if self.mode == 0:
            w = w.view(-1, 3, self.k)
        else:
            w = w.view(-1, self.k, 3)
        return w
######################################################################################################
class query_Attention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., k=16):
        super().__init__()
        self.num_heads = num_heads
        self.k2 = k*k
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Parameter(torch.ones((1, k**2, dim)), requires_grad=True)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = self.q.expand(B, -1, -1).view(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, self.k2, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class query_SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, k=16):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = query_Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, k=k)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Global_branch(nn.Module):
    def __init__(self, inp=3, oup=64, num_heads=2, k=16):
        super(Global_branch, self).__init__()
        self.k = k
        self.conv_embedding1 = Conv(inp, oup//2, kernal_size=3, stride=2, padding=1)
        self.conv_embedding2 = Conv(oup//2, oup, kernal_size=3, stride=2, padding=1)
        self.generator = query_SABlock(dim=oup, num_heads=num_heads, k=k)
        self.fc = nn.Linear(oup, 1)
        self.p_base = nn.Parameter(torch.eye(k), requires_grad=True)

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name == 'generator.attn.v.weight':
                nn.init.constant_(p, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.conv_embedding1(x)
        x = self.conv_embedding2(x)
        x = self.generator(x)
        x = self.fc(x)

        B, N, _ = x.shape
        x = x.squeeze(-1).view(B, self.k, self.k)
        x = x+self.p_base
        return x
################################################################################################################

class Domain_g(nn.Module):
    def __init__(self, inp=3, oup_g=64, num_heads=4, k=16, oup_l=[16, 64, 128, 256]):
        super(Domain_g, self).__init__()
        self.global_branch = Global_branch(inp=inp, oup=oup_g, num_heads=num_heads, k=k)
        self.local_branch1 = local_branch(inp=inp, oup=oup_l, k=k, mode=0)
        self.local_branch2 = local_branch(inp=k, oup=oup_l, k=k, mode=1)

    def forward(self, x):
        w1 = self.local_branch1(x).unsqueeze(1)
        w2 = self.global_branch(x).unsqueeze(1)
        x = x.permute(0, 2, 3, 1)

        x = x@w1
        x = x@w2
        x2 = x.permute(0, 3, 1, 2)
        w3 = self.local_branch2(x2).unsqueeze(1)
        x = x@w3
        x = x.permute(0, 3, 1, 2)
        x_tensor = x.clone()
        #normalize
        for i in range(3):
            xi = x_tensor[:, i]
            x_max = torch.max(xi)
            x_min = torch.min(xi)
            if i == 0:
                x[:, i] = (x_tensor[:, i]-x_min)/(x_max*random.uniform(0.9, 1.0))
            elif i == 1:
                x[:, i] = (x_tensor[:, i] - x_min) / (x_max*random.uniform(0.8, 0.9))
            else:
                x[:, i] = (x_tensor[:, i] - x_min) / (x_max*random.uniform(1.6, 1.7))

        return x

# def get_mean_std(img):
#     h, w, c = img.shape
#     for i in range(c):


if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    from torchvision.transforms import Compose, ToTensor, Resize
    img_path = "/home/chj/Desktop/S-UODAC2020/type7/000022.jpg"
    # mean = np.array([[[123.675, 116.28, 103.53]]], dtype=np.float32)
    # std = np.array([[[58.395, 57.12, 57.375]]], dtype=np.float32)
    img = cv2.imread(img_path)
    # img = (img-mean)/std
    # img = torch.from_numpy(img)
    # img = img.permute(2, 0, 1)
    transform = Compose([
        ToTensor(),
        Resize((1333, 800))
    ])
    x = transform(img).unsqueeze(0)

    net = Domain_g()
    out = net(x)
    print(out.shape)

    # out = out.squeeze(0).permute(1, 2, 0)
    # out = out.detach().numpy()
    # out = out*255
    # out = np.array(out, dtype=np.int)
    #
    # # x = (img * mean+std).permute(1, 2, 0)
    # # x = x.detach().numpy()
    # # x = np.array(x, dtype=np.int)
    #
    # # out = np.concatenate([out, x],axis=1)
    #
    # # print(out)
    # # plt.imshow(img[:, :, ::-1])
    # plt.imshow(out[:, :, ::-1])
    # plt.show()