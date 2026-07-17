import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from einops import rearrange
import numpy as np


def C(x):
    if x == 0:
        return 1 / np.sqrt(2)
    return 1.0


def initDCTKernel(N):
    kernel = np.zeros((N, N))
    for x in range(N):
        for y in range(N):
            kernel[x, y] = 2 / N * C(x) * C(y) * np.cos((2*x + 1)*y*np.pi/(2*N))
    kernel = kernel.astype(np.float32)
    return kernel


def initIDCTKernel(N):
    kernel = np.zeros((N, N))
    for x in range(N):
        for y in range(N):
            kernel[x, y] = 2 / N * C(x) * C(y) * np.cos((2*x + 1)*y*np.pi/(2*N))
    kernel = kernel.astype(np.float32)
    return kernel


class DCT(nn.Module):
    def __init__(self, ksz, in_channels):
        super(DCT, self).__init__()
        self.kernel_size = ksz

        # 初始化DCT核，假设initDCTKernel函数返回的是[ksz, ksz]形状的二维数组
        dct_kernel = initDCTKernel(self.kernel_size)
        dct_kernel = torch.Tensor(dct_kernel)
        # 扩展DCT核以匹配形状 [in_channels, 1, ksz, ksz]
        # 这里我们假设每个输入通道使用相同的DCT核
        self.in_kernel = dct_kernel.view(1, 1, self.kernel_size, self.kernel_size)
        self.in_kernel = self.in_kernel.repeat((in_channels, 1, 1, 1))

        # 将 self.in_kernel 转换为 nn.Parameter
        self.in_kernel = nn.Parameter(self.in_kernel)

        self.in_kernel.requires_grad = False

    def forward(self, x):
        # 应用卷积，不使用groups参数
        out = F.conv2d(x, self.in_kernel, padding=(self.kernel_size - 1) // 2, stride=1, groups=x.shape[1])
        # print("out:",out.shape)
        return out


class IDCT(nn.Module):
    def __init__(self,ksz, in_channels):
        super(IDCT, self).__init__()
        self.kernel_size = ksz
        out_kernel = initIDCTKernel(self.kernel_size)
        out_kernel = torch.Tensor(out_kernel)
        # 扩展维度
        out_kernel = out_kernel.unsqueeze(0).unsqueeze(0)  # 形状变为 [1, 1, ksz, ksz]

        # 扩展DCT核以匹配形状 [in_channels, 1, ksz, ksz]
        # 这里我们假设每个输入通道使用相同的DCT核

        self.out_kernel = out_kernel.view(1, 1, self.kernel_size, self.kernel_size)
        self.out_kernel = self.out_kernel.repeat((in_channels, 1, 1, 1))

        self.out_kernel = nn.Parameter(self.out_kernel)
        self.out_kernel.requires_grad = False

    def forward(self, x):
        out = F.conv2d(input=x, weight=self.out_kernel, padding=(self.kernel_size - 1) // 2, stride=1, groups=x.shape[1])
        return out


class Upsample(nn.Sequential):

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class MSAX(nn.Module):
    def __init__(self, dim, upscale):
        super().__init__()
        # (H/4,W/4)
        self.D0 = nn.Sequential(
            nn.Conv2d(dim, 156, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(156, 156, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(156, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        # (H/2,W/2)
        self.D1 = nn.Sequential(
            nn.Conv2d(dim, 156, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(156, 156, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(156, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.upsample1 = Upsample(upscale, dim)
        self.upsample2 = Upsample(upscale * 2, dim)
        self.act = nn.GELU()
        self.pointwise = nn.Conv2d(dim * 3, dim, kernel_size=1)
        self.depthwise = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3)

    def forward(self, x):
        x0 = self.D0(x)
        x1 = self.D1(x0)
        x = self.pointwise(self.act(self.depthwise(torch.cat([self.upsample1(x0), self.upsample2(x1), x], dim=1))))

        return x


class EP(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.canny = CannyEdgeDetector()
        self.res0 = ResBlock(dim, dim)
        self.leaky = nn.LeakyReLU(inplace=True)
        self.res1 = ResBlock(dim, dim)
        self.res2 = ResBlock(dim, dim)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_ = self.canny(x)
        x_0 = self.Sigmoid(self.leaky(self.res1(self.leaky(self.res0(x_)))))
        x_1 = self.res2(x_)
        return x_0 * x_1 + x - x_


class MSI_FSMM(nn.Module):

    def __init__(self, dim, upscale):
        super().__init__()
        self.MScale = MSAX(dim, upscale)
        self.FDSS = FDSS(dim)
        self.EP = EP(dim)

    def forward(self, x):

        x = self.MScale(x) + self.FDSS(x) + self.EP(x)

        return x


class HSI_FSMM(nn.Module):

    def __init__(self, dim, upscale):
        super().__init__()
        self.conv4 = nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, groups=4)
        self.conv16 = nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, groups=16)
        self.conv8 = nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, groups=8)
        self.conv1_1 = nn.Conv2d(dim // 2 * 3, dim // 2, 1)

        self.MScale = MSAX(dim // 2, upscale)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.leaky = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv2d(dim // 2 * 3, dim // 2, 1)
        self.FDCS = FDCS(dim // 2)

    def forward(self, x):
        x4 = self.conv4(x)
        x16 = self.conv16(x)
        x8 = self.conv8(x)
        x_0 = torch.cat((x4, x16, x8), dim=1)
        x_0 = self.conv1_1(x_0)
        x_0 = self.leaky(x_0)
        x = self.MScale(x_0) + self.FDCS(x_0)

        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True,
                      padding_mode="reflect"),
        )

    def forward(self, x):
        out = self.conv(x)
        return out + x


class ChannelAttention(nn.Module):

    def __init__(self, num_feat, squeeze_factor=30):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class FDSS(nn.Module):    # 频率域,

    def __init__(self, dim):
        super(FDSS, self).__init__()

        self.dct = DCT(ksz=3, in_channels=dim)
        self.idct = IDCT(ksz=3, in_channels=dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, padding_mode="reflect")
        self.leaky = nn.LeakyReLU(inplace=True)
        self.res = ResBlock(dim, dim)

    def forward(self, x):
        x_ = self.dct(x)
        y_spa = self.res(self.leaky(self.conv(x_)))
        y = self.idct(y_spa)
        return y+x


class FDCS(nn.Module):    # 频率域,

    def __init__(self, dim):
        super(FDCS, self).__init__()

        self.dct = DCT(ksz=3, in_channels=dim)
        self.idct = IDCT(ksz=3, in_channels=dim)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.res = ResBlock(dim, dim)

    def forward(self, x):
        x_ = self.dct(x)
        y_spe = self.sigmoid(self.conv1(self.pool(x_)))*self.res(x_)
        y = self.idct(y_spe)
        return y+x


class CannyEdgeDetector(nn.Module):
    def __init__(self):
        super(CannyEdgeDetector, self).__init__()
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device='cuda').view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device='cuda').view(1, 1, 3, 3)

    def forward(self, input):
        batch_size, num_channels, height, width = input.size()
        input = input.view(batch_size * num_channels, 1, height, width)

        grad_x = F.conv2d(input, self.sobel_x, padding=1)
        grad_y = F.conv2d(input, self.sobel_y, padding=1)

        magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        angle = torch.atan2(grad_y, grad_x)

        angle_quantized = torch.round(angle / (0.5 * 3.1416)) * 0.5 * 3.1416

        edge_strength = torch.zeros_like(magnitude)
        for i in range(1, 4):
            idx = (angle_quantized == i * 0.5 * 3.1416)
            edge_strength[idx & (magnitude > torch.roll(magnitude, shifts=1, dims=-1))] = 0
            edge_strength[idx & (magnitude > torch.roll(magnitude, shifts=-1, dims=-1))] = 0
        low_threshold = 0.1
        high_threshold = 0.3
        edge_map = torch.zeros_like(magnitude)
        edge_map[edge_strength > high_threshold] = 1
        edge_map[(edge_strength <= high_threshold) & (edge_strength >= low_threshold)] = 0.5

        edge_map = edge_map.view(batch_size, num_channels, height, width)

        return edge_map


class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)


# MLP块
class Mlp(nn.Module):

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


def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


###############------Triangular_Window_Partition------###############
def window_partition_triangular(x, window_size, masks):   #[1, 64, 64, 180]
    b, h, w, c = x.shape
    m = len(masks)
    ws = window_size
    h_ws = h // ws
    w_ws = w // ws
    x = x.view(b, h_ws, ws, w_ws, ws, c)    #b, h/ws, ws, w/ws, ws, c
    windows = x.permute(0, 1, 3, 5, 2, 4).contiguous().view(-1, ws, ws) #b, h/ws, w/ws, c, ws, ws-->b*(h_ws)*(w_ws)*c, ws, ws
    #window_mask=torch.zeros((len(masks), windows.shape[0], ws//2 * ws//2), dtype=windows.dtype).to(x.device)
    window_masks = []
    for mask in masks:
        mask = mask.expand(windows.shape[0], -1, -1)
        window_mask = windows[mask]
        window_masks.append(window_mask.unsqueeze(0))
    window_masks = torch.cat(window_masks, dim=0)
    window_masks = window_masks.view(m, windows.shape[0], -1)
    m, _, n = window_masks.shape
    window_masks = window_masks.view(m, -1, c, n).permute(1, 0, 3, 2).contiguous()  #[m, b*(h_ws)*(w_ws)*c, n]->[b*(h_ws)*(w_ws), m, n, c]
    return window_masks
###############------Triangular_Window_Partition------###############



###############------Triangular_Window_Reverse------###############
def window_reverse_triangular(x, window_size, masks):
    b_, m, n, c = x.shape   #[b*(h_ws)*(w_ws), m, n, c]
    x = x.permute(1, 0, 3, 2).contiguous().view(m, -1)  #[m, b*(h_ws)*(w_ws)*c, n]
    reconstructed = torch.zeros((b_*c, window_size, window_size), dtype=x.dtype).to(x.device)
    for mask, x_ in zip(masks, x):
        mask = mask.expand(b_*c, -1, -1)
        reconstructed[mask] = x_   #[b*(h_ws)*(w_ws)*c, ws, ws]
    return reconstructed
###############------Triangular_Window_Reverse------###############


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: b, h*w, c
        """
        h, w = self.input_resolution
        b, seq_len, c = x.shape
        assert seq_len == h * w, 'input feature has wrong size'
        assert h % 2 == 0 and w % 2 == 0, f'x size ({h}*{w}) are not even.'

        x = x.view(b, h, w, c)

        x0 = x[:, 0::2, 0::2, :]  # b h/2 w/2 c
        x1 = x[:, 1::2, 0::2, :]  # b h/2 w/2 c
        x2 = x[:, 0::2, 1::2, :]  # b h/2 w/2 c
        x3 = x[:, 1::2, 1::2, :]  # b h/2 w/2 c
        x = torch.cat([x0, x1, x2, x3], -1)  # b h/2 w/2 4*c
        x = x.view(b, -1, 4 * c)  # b h/2*w/2 4*c

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchEmbed(nn.Module):

    def __init__(self, in_chans, img_size=64, patch_size=1, embed_dim=48, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)

        if self.norm is not None:
            x = self.norm(x)

        return x


class PatchUnEmbed(nn.Module):

    def __init__(self, img_size=64, patch_size=1, in_chans=3, embed_dim=48):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).contiguous().reshape(x.shape[0], self.embed_dim, x_size[0], x_size[1])
        return x


class Spa_RWT(nn.Module):

    def __init__(self,
                 dim,
                 input_resolution,
                 window_size,
                 overlap_ratio,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 mlp_ratio=4,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        self.norm1 = norm_layer(dim)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size,
                                padding=(self.overlap_win_size - window_size) // 2)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((window_size + self.overlap_win_size - 1) * (window_size + self.overlap_win_size - 1),
                        num_heads))

        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)
        self.conv1 = nn.Conv2d(dim * 2, dim, 1)
        self.att = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, x_size, rpi):
        h, w = x_size
        b, _, c = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)
        y = self.norm1(y)
        y = y.view(b, h, w, c)

        kv_in = self.kv(x).reshape(b, h, w, 2, c).permute(3, 0, 4, 1, 2)
        q_in = self.q(y)
        kv = torch.cat((kv_in[0], kv_in[1]), dim=1)

        # partition windows
        q_windows = window_partition(q_in, self.window_size)  # nw*b, window_size, window_size, c
        q_windows = q_windows.view(-1, self.window_size * self.window_size, c)
        kv_windows = self.unfold(kv)
        kv_windows = rearrange(kv_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch', nc=2, ch=c,
                               owh=self.overlap_win_size, oww=self.overlap_win_size).contiguous()
        k_windows, v_windows = kv_windows[0], kv_windows[1]

        b_, nq, _ = q_windows.shape
        _, n, _ = k_windows.shape
        d = self.dim // self.num_heads
        q = q_windows.reshape(b_, nq, self.num_heads, d).permute(0, 2, 1, 3)  # 256,6,16,16
        k = k_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)  # 256,6,36,16
        v = v_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)  # nw*b, nH, n, d

        # windows attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size * self.window_size, self.overlap_win_size * self.overlap_win_size, -1)

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn_windows = (attn @ v).transpose(1, 2).reshape(b_, nq, self.dim)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.dim)
        x = window_reverse(attn_windows, self.window_size, h, w)
        x = x.view(b, h * w, self.dim)
        x = self.proj(x)

        att_ac = self.sigmoid(self.att(self.conv1(torch.cat((kv_in[0], q_in.permute(0,3,1,2)), dim=1))))    # 用kq做通道注意力图
        x = x.view(b, c, h, w) * att_ac + x.view(b, c, h, w)
        x = x.view(b, h*w, c) + shortcut

        x = x + self.mlp(self.norm2(x))

        return x


class Spe_RWT(nn.Module):

    def __init__(self,
                 dim,
                 input_resolution,
                 window_size,
                 overlap_ratio,
                 num_heads,
                 compress_ratio,
                 squeeze_factor,
                 qkv_bias=True,
                 qk_scale=None,
                 mlp_ratio=4,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size  # (4*0.5)+4=6  overlap的窗口尺寸

        self.norm1 = norm_layer(dim)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        # nn.Unfold 用于在输入的张量上执行滑动窗口操作
        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size,
                                padding=(self.overlap_win_size - window_size) // 2)

        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)
        self.conv_block = CAB(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)
        self.conv1 = nn.Conv2d(dim * 2, dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, x_size):
        h, w = x_size
        b, _, c = x.shape
        shortcut = y
        x = self.norm1(x)
        x = x.view(b, h, w, c)
        y = self.norm1(y)
        y = y.view(b, h, w, c)

        q_in = self.q(x)  # b, h, w, c
        k_in = self.k(y)
        v_in = self.v(y)
        qk = torch.cat((q_in, k_in), dim=-1)  # b, h, w, 2*c
        qk = qk.permute(0, 3, 1, 2)

        # partition windows
        v_windows = window_partition(v_in, self.window_size)  # nw*b, window_size, window_size, c
        v_windows = v_windows.view(-1, self.window_size * self.window_size, c)  # （256, 16, 48）
        qk_windows = self.unfold(qk)  # b, c*w*w, nw
        qk_windows = rearrange(qk_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch', nc=2, ch=c,
                               owh=self.overlap_win_size, oww=self.overlap_win_size).contiguous()
        q_windows, k_windows = qk_windows[0], qk_windows[1]
        b_, nq, _ = v_windows.shape
        _, n, _ = k_windows.shape
        d = self.dim // self.num_heads
        q = q_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)  # 256,4,36,12
        k = k_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)  # 256,4,36,12
        v = v_windows.reshape(b_, nq, self.num_heads, d).permute(0, 2, 1, 3)  # 256,4,16,12

        # windows attention
        q = q * self.scale
        attn = (q.transpose(-2, -1) @ k)
        attn = self.softmax(attn)  # (256, 4, 12, 12)
        attn_windows = (v @ attn).transpose(1, 2).reshape(b_, nq, self.dim)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.dim)  # 256,4,4,96
        x = window_reverse(attn_windows, self.window_size, h, w)  # b h w c
        x = x.view(b, h * w, self.dim)
        x = self.proj(x)

        kq = torch.cat((k_in, q_in), dim=-1)
        att_ac = self.sigmoid((self.conv1(kq.permute(0,3,1,2))))   # 用kq做空间注意力图
        x = x.view(b, c, h, w) * att_ac + x.view(b, c, h, w)   # 在注意力内部加一个跳跃连接
        x = x.view(b, h*w, c) + shortcut

        x_ = self.norm2(x)
        x_ = x_.view(b, h, w, c)
        conv_x = self.conv_block(x_.permute(0, 3, 1, 2))
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        x = conv_x + x

        return x


class Spa_TWT(nn.Module):

    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 mlp_ratio=4,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.norm1 = norm_layer(dim)
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)

        self.conv1 = nn.Conv2d(dim * 2, dim, 1)
        self.att = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, x_size, rpi, triangular_masks):
        h, w = x_size
        b, _, c = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        y = self.norm1(y)
        y = y.view(b, h, w, c)

        shifted_x = x  # [1, 64, 64, 180]
        shifted_y = y
        attn_mask = None
        #######----Cyclic_Shift + Mask----########

        #######----Partition_Windows----########

        x_windows = window_partition_triangular(shifted_x, 2 * self.window_size,
                                                triangular_masks)  # nw*b, window_size, window_size, c #[1, 64, 64, 180]->[16, 16, 16, 180]
        y_windows = window_partition_triangular(shifted_y, 2 * self.window_size,
                                                triangular_masks)  # nw*b, window_size, window_size, c #[1, 64, 64, 180]->[16, 16, 16, 180]
        _, m, n, _ = x_windows.shape  # [b*(h_ws)*(w_ws), m, n, c]
        x_windows = x_windows.view(-1, n, c)  # [b*(h_ws)*(w_ws)*m, n, c]  #[16, 16*16, 180]
        y_windows = y_windows.view(-1, n, c)  # [b*(h_ws)*(w_ws)*m, n, c]  #[16, 16*16, 180]
        #######----Partition_Windows----########

        #######----W-MSA/SW-MSA----########
        # attn_windows = self.attn(x_windows, rpi=rpi, mask=attn_mask)  # [16, 256, 180]
        # 开始计算QKV然后计算注意力
        b_, n, c = x_windows.shape  # [16, 256, 180]

        ##########------q, k, v------##########
        qk = self.qk(x_windows).reshape(b_, n, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1,
                                                                                       4)  # [16, 256, 540]->[3, 16, 6, 256, 30]
        v = self.v(y_windows).reshape(b_, n, 1, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qk[0], qk[1], v[0]  # make torchscript happy (cannot use tensor as tuple) #[16, 6, 256, 30]
        ##########------q, k, v------##########

        ##########------q*k------##########
        q = q * self.scale  # scale=0.18257418583505536 #[16, 6, 256, 30]
        attn = (q @ k.transpose(-2, -1))  # [16, 6, 256, 256]
        ##########------q*k------##########

        ##########--------Relative_Position_Bias--------##########
        relative_position_bias = self.relative_position_bias_table[rpi['rpi_rt'].view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        # Wh*Ww,Wh*Ww,nH #[256, 256, 6]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww #[6, 256, 256]
        attn = attn + relative_position_bias.unsqueeze(0)  # [16, 6, 256, 256]+[1, 6, 256, 256]=[16, 6, 256, 256]
        ##########--------Relative_Position_Bias--------##########
        attn = self.softmax(attn)
        ##########------masking+softmax------##########

        # attn = self.attn_drop(attn)  # [16, 6, 256, 256]

        ##########------(qk)*v+Linear------##########
        attn_windows = (attn @ v).transpose(1, 2).reshape(b_, n,
                                               c)  # [16, 6, 256, 256]*[16, 6, 256, 30]=[16, 6, 256, 30]->[16, 256, 180]
        attn_windows = self.proj(attn_windows)  # [16, 256, 180]
        ##########------(qk)*v+Linear------##########

        # attn_windows = self.proj_drop(attn_windows)  # [16, 256, 180]


        #######----Merge_Windows----########
        attn_windows = attn_windows.view(-1, m, n, c)  # [b*(h_ws)*(w_ws), m, n, c]
        shifted_x = window_reverse_triangular(attn_windows, 2 * self.window_size,
                                              triangular_masks)  # nw*b, window_size, window_size, c #[1, 64, 64, 180]->[16, 16, 16, 180]
        shifted_x = shifted_x.view(b, h // (2 * self.window_size), w // (2 * self.window_size), c,
                                   2 * self.window_size, 2 * self.window_size)
        shifted_x = shifted_x.permute(0, 1, 4, 2, 5, 3).contiguous().view(b, h, w, c)  # [1, 64, 64, 180]

        attn_x = shifted_x
        #######----Reverse_Cyclic_Shift----########

        attn_x = attn_x.view(b, h * w, c)  # [1, 64, 64, 180]->[1, 4096, 180]

        x = self.proj(attn_x)

        x = x + shortcut

        x = x + self.mlp(self.norm2(x))

        return x


class Spe_TWT(nn.Module):

    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 mlp_ratio=4,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.norm1 = norm_layer(dim)
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)

        self.conv1 = nn.Conv2d(dim * 2, dim, 1)
        self.att = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, x_size, triangular_masks):
        h, w = x_size
        b, _, c = x.shape
        shortcut = y
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        y = self.norm1(y)
        y = y.view(b, h, w, c)

        shifted_x = x  # [1, 64, 64, 180]
        shifted_y = y
        attn_mask = None
        #######----Cyclic_Shift + Mask----########

        #######----Partition_Windows----########

        x_windows = window_partition_triangular(shifted_x, 2 * self.window_size,
                                                triangular_masks)  # nw*b, window_size, window_size, c #[1, 64, 64, 180]->[16, 16, 16, 180]
        y_windows = window_partition_triangular(shifted_y, 2 * self.window_size,
                                                triangular_masks)  # nw*b, window_size, window_size, c #[1, 64, 64, 180]->[16, 16, 16, 180]
        _, m, n, _ = x_windows.shape  # [b*(h_ws)*(w_ws), m, n, c]
        x_windows = x_windows.view(-1, n, c)  # [b*(h_ws)*(w_ws)*m, n, c]  #[16, 16*16, 180]
        y_windows = y_windows.view(-1, n, c)  # [b*(h_ws)*(w_ws)*m, n, c]  #[16, 16*16, 180]
        #######----Partition_Windows----########

        #######----W-MSA/SW-MSA----########
        # attn_windows = self.attn(x_windows, rpi=rpi, mask=attn_mask)  # [16, 256, 180]
        # 开始计算QKV然后计算注意力
        b_, n, c = x_windows.shape  # [16, 256, 180]

        ##########------q, k, v------##########
        qk = self.qk(y_windows).reshape(b_, n, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1,
                                                                                               4)  # [16, 256, 540]->[3, 16, 6, 256, 30]
        v = self.v(x_windows).reshape(b_, n, 1, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qk[0], qk[1], v[0]  # make torchscript happy (cannot use tensor as tuple) #[16, 6, 256, 30]
        ##########------q, k, v------##########

        ##########------q*k------##########
        q = q * self.scale  # scale=0.18257418583505536 #[16, 6, 256, 30]
        attn = (q.transpose(-2, -1) @ k)  # [16, 6, 30, 30]
        ##########------q*k------##########

        attn = self.softmax(attn)
        ##########------masking+softmax------##########

        # attn = self.attn_drop(attn)  # [16, 6, 256, 256]

        ##########------(qk)*v+Linear------##########
        attn_windows = (v @ attn).transpose(1, 2).reshape(b_, n,
                                                          c)  # [16, 6, 256, 30]*[16, 6, 30, 30]=[16, 6, 256, 30]->[16, 256, 180]
        attn_windows = self.proj(attn_windows)  # [16, 256, 180]
        ##########------(qk)*v+Linear------##########

        # attn_windows = self.proj_drop(attn_windows)  # [16, 256, 180]

        #######----Merge_Windows----########
        attn_windows = attn_windows.view(-1, m, n, c)  # [b*(h_ws)*(w_ws), m, n, c]
        shifted_x = window_reverse_triangular(attn_windows, 2 * self.window_size,
                                              triangular_masks)  # nw*b, window_size, window_size, c #[1, 64, 64, 180]->[16, 16, 16, 180]
        shifted_x = shifted_x.view(b, h // (2 * self.window_size), w // (2 * self.window_size), c,
                                   2 * self.window_size, 2 * self.window_size)
        shifted_x = shifted_x.permute(0, 1, 4, 2, 5, 3).contiguous().view(b, h, w, c)  # [1, 64, 64, 180]

        attn_x = shifted_x
        #######----Reverse_Cyclic_Shift----########

        attn_x = attn_x.view(b, h * w, c)  # [1, 64, 64, 180]->[1, 4096, 180]

        x = self.proj(attn_x)

        x = x + shortcut

        x = x + self.mlp(self.norm2(x))

        return x


class BLOCK(nn.Module):

    def __init__(self,
                 compress_ratio,
                 squeeze_factor,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size,
                 overlap_ratio,
                 qkv_bias=True,
                 qk_scale=None,
                 mlp_ratio=4,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.Spa_RWT = Spa_RWT(
            dim=dim,
            input_resolution=input_resolution,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer
        )

        self.Spe_RWT = Spe_RWT(
            dim=dim,
            input_resolution=input_resolution,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            num_heads=num_heads,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer
        )

    def forward(self, x, y, x_size, params):
        x = self.Spa_RWT(x, y, x_size, params)
        y = self.Spe_RWT(x, y, x_size)

        return x, y


class BLOCK0(nn.Module):

    def __init__(self,
                 compress_ratio,
                 squeeze_factor,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size,
                 overlap_ratio,
                 qkv_bias=True,
                 qk_scale=None,
                 mlp_ratio=4,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.Spa_RWT = Spa_RWT(
            dim=dim,
            input_resolution=input_resolution,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer
        )

        self.Spe_RWT = Spe_RWT(
            dim=dim,
            input_resolution=input_resolution,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            num_heads=num_heads,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer
        )

    def forward(self, x, x_size, params):
        x0 = self.Spa_RWT(x, x, x_size, params)
        x = self.Spe_RWT(x0, x0, x_size)

        return x


class LAYER(nn.Module):

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_sizes,  # 修改为 window_sizes 列表
                 window_size0,
                 compress_ratio,
                 squeeze_factor,
                 overlap_ratio):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.num_heads = num_heads
        self.compress_ratio = compress_ratio
        self.squeeze_factor = squeeze_factor
        self.overlap_ratio = overlap_ratio

        assert len(window_sizes) == depth, "The length of window_sizes must match depth."
        self.window_sizes = window_sizes  # 保存不同的 window_sizes

        self.blocks = nn.ModuleList([
            BLOCK(
                compress_ratio,
                squeeze_factor,
                dim,
                input_resolution,
                num_heads,
                window_size=window_sizes[i],  # 每个 BLOCK 的窗口大小不同
                overlap_ratio=overlap_ratio,
                qkv_bias=True,
                qk_scale=None,
                mlp_ratio=4,
                norm_layer=nn.LayerNorm
            ) for i in range(depth)
        ])

        self.Spa_TWT = Spa_TWT(dim, window_size0, num_heads)
        self.Spe_TWT = Spe_TWT(dim, window_size0, num_heads)

    def forward(self, x, y, x_size, params, paramsRT, triangular_masks):
        for i, blk in enumerate(self.blocks):
            # 每个块传入对应的params元素
            block_params = params['rpi'][i]
            x, y = blk(x, y, x_size,  block_params)

        x = self.Spa_TWT(x, y, x_size, paramsRT, triangular_masks)
        y = self.Spe_TWT(x, y, x_size, triangular_masks)

        return x, y


class MS2WT(nn.Module):

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 window_size0,
                 compress_ratio,
                 squeeze_factor,
                 overlap_ratio,
                 img_size=64,
                 patch_size=1,
                 resi_connection='1conv'):
        super(MS2WT, self).__init__()

        self.layer = LAYER(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_sizes=window_size,
            window_size0=window_size0,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            overlap_ratio=overlap_ratio
        )

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == 'identity':
            self.conv = nn.Identity()

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=dim, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=dim, embed_dim=dim)

    def forward(self, x, y, x_size, params, paramsRT, triangular_masks):

        x0, y0 = self.layer(x, y, x_size, params, paramsRT, triangular_masks)
        x = self.patch_embed(self.conv(self.patch_unembed(x0, x_size))) + x
        y = self.patch_embed(self.conv(self.patch_unembed(y0, x_size))) + y

        return x, y


class DDI(nn.Module):

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size,
                 compress_ratio,
                 squeeze_factor,
                 overlap_ratio,
                 img_size=64,
                 patch_size=1,
                 resi_connection='1conv'):
        super(DDI, self).__init__()

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == 'identity':
            self.conv = nn.Identity()

        self.block = BLOCK0(
            compress_ratio,
            squeeze_factor,
            dim,
            input_resolution,
            num_heads,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            qkv_bias=True,
            qk_scale=None,
            mlp_ratio=4,
            norm_layer=nn.LayerNorm
        )

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=dim, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=dim, embed_dim=dim)

    def forward(self, x, x_size, params):

        x0 = self.block(x, x_size, params['rpi'])
        x = self.patch_embed(self.conv(self.patch_unembed(x0, x_size))) + x

        return x


@ARCH_REGISTRY.register()
class FEM2Transformer(nn.Module):

    def __init__(self,
                 scale_ratio,
                 n_select_bands,
                 n_bands,
                 upscale=4,
                 embed_dim=48,
                 img_size=64,
                 patch_size=1,
                 depths=3,
                 num_heads=4,
                 window_size4=4,
                 window_size=(4, 4, 8),
                 window_size0=16,
                 compress_ratio=3,
                 squeeze_factor=30,
                 overlap_ratio=0.5,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 img_range=1.,
                 resi_connection='1conv'):

        super(FEM2Transformer, self).__init__()

        self.upscale = upscale
        self.window_size4 = window_size4
        self.window_size = window_size
        self.window_size0 = window_size0
        self.overlap_ratio = overlap_ratio
        self.scale_ratio = scale_ratio
        self.n_bands = n_bands
        self.n_select_bands = n_select_bands
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.img_range = img_range

        self.weight = nn.Parameter(torch.tensor([0.5]))
        self.norm = nn.LayerNorm(embed_dim)
        self.Conv5_dim = nn.Conv2d(in_channels=n_select_bands, out_channels=embed_dim, kernel_size=3, padding=(1, 1))
        self.Convin_dim = nn.Conv2d(in_channels=n_bands, out_channels=embed_dim * 2, kernel_size=3, padding=(1, 1))
        self.depthwise = nn.Conv2d(embed_dim * 3, embed_dim * 3, 3, 1, 1, groups=embed_dim * 3)
        self.act = nn.GELU()
        self.pointwise = nn.Linear(embed_dim * 3, embed_dim)
        self.mean = torch.zeros(1, 1, 1, 1)
        self.MSI_FSMM = MSI_FSMM(embed_dim, upscale)
        self.HSI_FSMM = HSI_FSMM(embed_dim * 2, upscale)
        self.FDSS = FDSS(embed_dim)
        self.FDCS = FDCS(embed_dim)

        num_feat = 64  # Only used once in the final reconstruction module

        # 为每个窗口大小计算相对位置编码
        self.relative_position_index = self.calculate_multiple_rpi_oca(window_size)    # 这返回值是一个列表
        self.relative_position_index_rt = self.calculate_rpi_sa()

        relative_position_index_4 = self.calculate_rpi_oca(window_size4)
        self.register_buffer('relative_position_index_4', relative_position_index_4)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim)

        self.MS2WT = MS2WT(
            dim=embed_dim,
            input_resolution=(patches_resolution[0], patches_resolution[1]),
            depth=depths,
            num_heads=num_heads,
            window_size=window_size,
            window_size0=window_size0,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            overlap_ratio=overlap_ratio,
            img_size=img_size,
            patch_size=patch_size,
            resi_connection=resi_connection
        )
        self.DDI0 = DDI(
            dim=embed_dim,
            input_resolution=(patches_resolution[0], patches_resolution[1]),
            num_heads=num_heads,
            window_size=window_size4,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            overlap_ratio=overlap_ratio,
            img_size=img_size,
            patch_size=patch_size,
            resi_connection=resi_connection
        )

        self.DDI1 = DDI(
            dim=embed_dim,
            input_resolution=(patches_resolution[0], patches_resolution[1]),
            num_heads=num_heads,
            window_size=window_size4,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            overlap_ratio=overlap_ratio,
            img_size=img_size,
            patch_size=patch_size,
            resi_connection=resi_connection
        )
        # Reconstruction module: convolution + activation + convolution
        self.conv_before = nn.Sequential(
            nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))  # num_feat=64 (通道数从48到64)

        self.conv_last = nn.Conv2d(num_feat, n_bands, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def calculate_rpi_oca(self, window_size):
        """
        根据给定的窗口大小计算相对位置索引。
        """
        window_size_ori = window_size
        window_size_ext = window_size + int(self.overlap_ratio * window_size)

        coords_h = torch.arange(window_size_ori)
        coords_w = torch.arange(window_size_ori)
        coords_ori = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_ori_flatten = torch.flatten(coords_ori, 1)

        coords_h = torch.arange(window_size_ext)
        coords_w = torch.arange(window_size_ext)
        coords_ext = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_ext_flatten = torch.flatten(coords_ext, 1)

        relative_coords = coords_ext_flatten[:, None, :] - coords_ori_flatten[:, :, None]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size_ori - window_size_ext + 1
        relative_coords[:, :, 1] += window_size_ori - window_size_ext + 1
        relative_coords[:, :, 0] *= window_size_ori + window_size_ext - 1
        relative_position_index0 = relative_coords.sum(-1)
        return relative_position_index0

    def calculate_multiple_rpi_oca(self, window_sizes):
        """
        计算多个窗口大小对应的相对位置编码。
        """
        rpi_list = []
        for window_size in window_sizes:
            rpi_list.append(self.calculate_rpi_oca(window_size))

        return rpi_list

    def triangle_masks(self, x):
        ws = 2 * self.window_size0
        rows = torch.arange(ws).unsqueeze(1).repeat(1, ws)
        cols = torch.arange(ws).unsqueeze(0).repeat(ws, 1)

        upper_triangle_mask = (cols > rows) & (rows + cols < ws)
        right_triangle_mask = (cols >= rows) & (rows + cols >= ws)
        bottom_triangle_mask = (cols < rows) & (rows + cols >= ws - 1)
        left_triangle_mask = (cols <= rows) & (rows + cols < ws - 1)

        return [upper_triangle_mask.to(x.device), right_triangle_mask.to(x.device), bottom_triangle_mask.to(x.device),
                left_triangle_mask.to(x.device)]

    def calculate_rpi_sa(self):
        coords_h = torch.arange(self.window_size0)
        coords_w = torch.arange(self.window_size0)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size0 - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size0 - 1
        relative_coords[:, :, 0] *= 2 * self.window_size0 - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        return relative_position_index

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, y):
        x_size = (x.shape[2], x.shape[3])

        params = {'rpi': self.relative_position_index}
        paramsRT = {'rpi_rt': self.relative_position_index_rt}

        triangular_masks = tuple(self.triangle_masks(x))

        x = self.patch_embed(x)
        y = self.patch_embed(y)

        x, y = self.MS2WT(x, y, x_size, params, paramsRT, triangular_masks)

        x = self.norm(x)
        y = self.norm(y)
        x = self.patch_unembed(x, x_size)
        y = self.patch_unembed(y, x_size)

        return x, y

    def forward_features0(self, x):
        x_size = (x.shape[2], x.shape[3])

        params = {'rpi': self.relative_position_index_4}

        x = self.patch_embed(x)

        x = self.DDI0(x, x_size, params)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)

        return x

    def forward_features1(self, x):
        x_size = (x.shape[2], x.shape[3])

        params = {'rpi': self.relative_position_index_4}

        x = self.patch_embed(x)

        x = self.DDI1(x, x_size, params)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, LR_HSI, HR_MSI):
        lms = F.interpolate(LR_HSI, scale_factor=self.scale_ratio, mode='bilinear')

        x = self.Conv5_dim(HR_MSI)
        y = self.Convin_dim(lms)  # x是MSI，y是HSI

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        self.mean = self.mean.type_as(y)
        y = (y - self.mean) * self.img_range

        x = self.forward_features0(self.MSI_FSMM(x))
        y = self.forward_features1(self.HSI_FSMM(y))

        x0, y0 = self.forward_features(x, y)
        z = torch.add(x0, y0)
        z1 = self.FDSS(z)
        z2 = self.FDCS(z)
        z = z1+z2+z

        x = self.conv_before(z)
        x = self.conv_last(x)
        x = x + lms

        x = x / self.img_range + self.mean

        return x


# 兼容别名：旧权重（完整模型 pickle）可正常加载
FS_MS2WT = FEM2Transformer
