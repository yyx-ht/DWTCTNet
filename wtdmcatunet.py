import math
import warnings
import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
from wtconv import DepthwiseSeparableConvWithWTConv2d
import torch.nn.functional as F
from wtdown import Down_wt

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x,H,W

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows
def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class window_attenblock(nn.Module):
    def __init__(self,dim,window_size,num_heads,qkv_bias=True,attn_drop =0,proj_drop=0,sr_ratio=1):
        super(window_attenblock,self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale =  head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        self.proj3 = nn.Linear(dim, dim)


        self.proj_drop = nn.Dropout(proj_drop)
        self.q1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.q2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.q3 = nn.Linear(dim, dim, bias=qkv_bias)

        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w],indexing = 'ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1,x2,x3,x4,H,W):
        B, N, C = x1.shape
        assert N == H * W, "input feature has wrong size"

        x1 = x1.view(B, H, W, C)
        x2 = x2.view(B, H, W, C)
        x3 = x3.view(B, H, W, C)
        x4 = x4.view(B, H, W, C)


        x_windows1 = window_partition(x1, self.window_size)
        x_windows2 = window_partition(x2, self.window_size)
        x_windows3 = window_partition(x3, self.window_size)
        x_windows4 = window_partition(x4, self.window_size)


        x_windows1 = x_windows1.view(-1, self.window_size * self.window_size, C)
        x_windows2 = x_windows2.view(-1, self.window_size * self.window_size, C)
        x_windows3 = x_windows3.view(-1, self.window_size * self.window_size, C)
        x_windows4 = x_windows4.view(-1, self.window_size * self.window_size, C)
        B_,N_,_ = x_windows1.shape


        q1 = self.q1(x_windows2).reshape(B_, N_, self.num_heads, C // self.num_heads).permute( 0, 2, 1, 3)
        q2 = self.q2(x_windows3).reshape(B_, N_, self.num_heads, C // self.num_heads).permute( 0, 2, 1, 3)
        q3 = self.q3(x_windows4).reshape(B_, N_, self.num_heads, C // self.num_heads).permute( 0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x1.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            k = self.k(x_).reshape(B_, N_, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            v = self.v(x_).reshape(B_, N_, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
           k = self.k(x_windows1).reshape(B_, N_, self.num_heads, C // self.num_heads).permute(0, 2, 1,3)
           v = self.v(x_windows1).reshape(B_, N_, self.num_heads, C // self.num_heads).permute(0, 2, 1,3)

        q1 = q1 * self.scale
        q2 = q2 * self.scale
        q3 = q3 * self.scale


        # 计算注意力分数
        attn1 = self.softmax(q1 @ k.transpose(-2, -1))
        attn2 = self.softmax(q2 @ k.transpose(-2, -1))
        attn3 = self.softmax(q3 @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn1 = attn1 + relative_position_bias.unsqueeze(0)
        attn2 = attn2 + relative_position_bias.unsqueeze(0)
        attn3 = attn3 + relative_position_bias.unsqueeze(0)


        # 计算注意力加权后的输出
        x1 = (self.attn_drop(attn1) @ v).transpose(1, 2).reshape(B_, N_, C)
        x2 = (self.attn_drop(attn2) @ v).transpose(1, 2).reshape(B_, N_, C)
        x3 = (self.attn_drop(attn3) @ v).transpose(1, 2).reshape(B_, N_, C)

        x1 = window_reverse(x1.view(-1, self.window_size, self.window_size, C),self.window_size,H,W)
        x2 = window_reverse(x2.view(-1, self.window_size, self.window_size, C),self.window_size,H,W)
        x3 = window_reverse(x3.view(-1, self.window_size, self.window_size, C),self.window_size,H,W)



        x1 = self.proj_drop(self.proj1(x1))
        x2 = self.proj_drop(self.proj2(x2))
        x3 = self.proj_drop(self.proj3(x3))


        return x1,x2,x3

class attblock(nn.Module):
    def __init__(self,indim,dim,act_layer=nn.GELU,drop=0,window_size=7,num_heads=4,drop_path=0,stride=2,sr_ratio=1,):
        super(attblock,self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')

        self.norm1_1 = nn.LayerNorm(dim)
        self.norm1_2 = nn.LayerNorm(dim)
        self.norm1_3 = nn.LayerNorm(dim)
        self.norm1_4 = nn.LayerNorm(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2_1 = nn.LayerNorm(dim)
        self.norm2_2 = nn.LayerNorm(dim)
        self.norm2_3 = nn.LayerNorm(dim)
        self.norm2_4 = nn.LayerNorm(dim)
        self.atten = window_attenblock(dim,window_size,num_heads,sr_ratio)

        self.mlp1 = Mlp(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=drop)
        self.mlp3 = Mlp(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=drop)
        self.mlp4 = Mlp(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=drop)

        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=stride, in_chans=indim, embed_dim=dim)
        self.patch_embed2 = OverlapPatchEmbed(patch_size=7, stride=stride, in_chans=indim, embed_dim=dim)
        self.patch_embed3 = OverlapPatchEmbed(patch_size=7, stride=stride, in_chans=indim, embed_dim=dim)
        self.patch_embed4 = OverlapPatchEmbed(patch_size=7, stride=stride, in_chans=indim, embed_dim=dim)
        self.out = nn.Conv2d(3*dim,dim,1,1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward (self,x):
        yL, yH = self.wt(x)
        B,_,_,_=yL.shape
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        yL,H,W = self.patch_embed1(yL)
        y_HL,H,W = self.patch_embed2(y_HL)
        y_LH,H,W = self.patch_embed3(y_LH)
        y_HH,H,W = self.patch_embed4(y_HH)
        f1 = yL
        f2 = y_HL
        f3 = y_LH
        f4 = y_HH
        C = f3.shape[-1]

        x1, x2, x3 = self.atten(self.norm1_1(yL), self.norm1_2(y_HL), self.norm1_3(y_LH), self.norm1_4(y_HH), H, W)

        x1 = x1.view(B, H * W, C)
        x2 = x2.view(B, H * W, C)
        x3 = x3.view(B, H * W, C)

        x1 = f2 + self.drop_path(x1)
        x2 = f3 + self.drop_path(x2)
        x3 = f4 + self.drop_path(x3)

        x1 = x1 + self.drop_path(self.mlp1(self.norm2_1(x1)))
        x2 = x2 + self.drop_path(self.mlp2(self.norm2_2(x2)))
        x3 = x3 + self.drop_path(self.mlp3(self.norm2_3(x3)))

        x1 = x1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x2 = x2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x3 = x3.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = self.out(torch.cat([x1,x2,x3],dim=1))

        return out
class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None


        self.apply(self._init_weights)

    def _init_weights(self, m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Res_block1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_block1, self).__init__()
        self.conv1 = DepthwiseSeparableConvWithWTConv2d(in_channels, out_channels, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = DepthwiseSeparableConvWithWTConv2d(out_channels, out_channels, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None


        self.apply(self._init_weights)


    def _init_weights(self, m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out



class CCA(nn.Module):
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, g, x):
        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d(g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g) / 2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out

class DMC_fusion(nn.Module):
    def __init__(self, in_channels ):
        super(DMC_fusion, self).__init__()
        self.in_channels = in_channels[0]
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(in_channels[3], in_channels[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[2]),
            nn.ReLU(inplace=True))
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels[2], in_channels[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[2]),
            nn.ReLU(inplace=True))
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(in_channels[2], in_channels[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[1]),
            nn.ReLU(inplace=True))
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels[1], in_channels[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[1]),
            nn.ReLU(inplace=True))
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels[1], in_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[0]),
            nn.ReLU(inplace=True))
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels[0], in_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[0]),
            nn.ReLU(inplace=True))
        self.conv1_0 = nn.Conv2d(in_channels[0], in_channels[0], 1, bias=False)
        self._init_conv1_0()
    def _init_conv1_0(self):
        eye = torch.eye(self.in_channels).view(self.in_channels, self.in_channels, 1, 1)
        self.conv1_0.weight.data = eye

    def forward(self, x1, x2, x3, x4):
        x4_1 = x4
        x4_2 = F.interpolate(self.conv4_2(x4_1), scale_factor=2)
        x3_1 = x4_2 * (self.conv3_1(x3))
        x3_2 = F.interpolate(self.conv3_2(x3_1), scale_factor=2)
        x2_1 = x3_2 * (self.conv2_1(x2))
        x2_2 = F.interpolate(self.conv2_2(x2_1), scale_factor=2)
        x1_1 = x2_2 * (self.conv1_1(x1))
        x1_1 = self.conv1_0(x1_1)

        return x1_1,x2_1, x3_1, x4_1
class wtdmcatunet(nn.Module):
    def __init__(self,inchanel,numclass,windowsize = [7,7],numhead=[4,8,16],stride=[1,1],sr_ratio=[8,4,2],drop_path_rate=0.1,dim =[64,128,256,512]):
        super(wtdmcatunet, self).__init__()
        self.inchanel = inchanel
        self.numclass = numclass

        self.convl_1 = Res_block1(inchanel,dim[0])
        self.convl_2 = Res_block1(dim[0], dim[1])
        self.convl_3 = Res_block1(dim[1], dim[2])
        self.convl_4 = Res_block1(dim[2], dim[3])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 4)]
        cur = 0

        self.att1 = attblock(inchanel,dim[0],window_size=windowsize[0],num_heads=numhead[0],stride=stride[0],sr_ratio=sr_ratio[0],drop_path=dpr[cur+0])
        self.att2 = attblock(dim[0], dim[1], window_size=windowsize[0],num_heads=numhead[1],stride=stride[0],sr_ratio=sr_ratio[1],drop_path=dpr[cur+1])
        self.att3 = attblock(dim[1], dim[2], window_size=windowsize[1],num_heads=numhead[2],stride=stride[0],sr_ratio=sr_ratio[2],drop_path=dpr[cur+2])

        self.lfuse1 = CCA(dim[0],dim[0])
        self.lfuse2 = CCA(dim[1],dim[1])
        self.lfuse3 = CCA(dim[2],dim[2])

        self.fuse = DMC_fusion(dim)

        self.down1 = Down_wt(dim[0],dim[0])
        self.down2 = Down_wt(dim[1],dim[1])
        self.down3 = Down_wt(dim[2],dim[2])

        # self.down1 = nn.MaxPool2d(2, 2)
        # self.down2 = nn.MaxPool2d(2, 2)
        # self.down3 = nn.MaxPool2d(2, 2)

        self.convr_4 = Res_block(dim[3], dim[2])
        self.convr_3 = Res_block(dim[2]+dim[2], dim[1])
        self.convr_2 = Res_block(dim[1]+dim[1], dim[0])
        self.convr_1 = Res_block(dim[0]+dim[0], dim[0])

        self.out = nn.Conv2d(dim[0],numclass,1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward (self,x):

        xl_11 = self.att1(x)#(4,64,56,56)
        xl_12 = self.convl_1(x)#(4,64,224,224)
        sl_12down = self.down1(xl_12)
        xl_1 = self.lfuse1(xl_11,sl_12down)#(4,64,112,112)

        xl_21 = self.att2(xl_1)
        xl_22 = self.convl_2(xl_1)#(4,128,112,112)
        sl_22down = self.down2(xl_22)
        xl_2 = self.lfuse2(xl_21,sl_22down)#(4,128,56,56)

        xl_31 = self.att3(xl_2)
        xl_32 = self.convl_3(xl_2)#(4,256,56,56)
        sl_32down = self.down3(xl_32)
        xl_3 = self.lfuse3(xl_31,sl_32down)#(4,256,28,28)
        xl_42 = self.convl_4(xl_3)#(4,256,28,28)
        x1,x2,x3,x4 = self.fuse(xl_12,xl_22,xl_32,xl_42)

        up_xr4 = F.interpolate(x4, size=xl_42.size()[2:], mode='bilinear', align_corners=False)#(4,512,28,28)
        xr4 = self.convr_4(up_xr4)
        up_xr3 = F.interpolate(xr4, size=xl_32.size()[2:], mode='bilinear', align_corners=False)#(4,256,56,56)
        xr3 = self.convr_3(torch.cat([up_xr3, x3], dim=1))
        up_xr2 = F.interpolate(xr3, size=xl_22.size()[2:], mode='bilinear', align_corners=False)#(4,128,112,112)
        xr2 = self.convr_2(torch.cat([up_xr2, x2], dim=1))#(4,64,112,112)
        up_xr1 = F.interpolate(xr2, size=xl_12.size()[2:], mode='bilinear', align_corners=False)
        xr1 = self.convr_1(torch.cat([up_xr1,x1], dim=1))#(4,32,224,224)
        out = self.out(xr1)
        return out





if __name__ == '__main__':
    model = wtdmcatunet(3,2)
    input_tensor = torch.ones((4, 3, 224, 224))
    output = model(input_tensor)
    print(output.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params}")