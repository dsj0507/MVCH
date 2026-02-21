import einops
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import torch.fft
from einops import rearrange, repeat, einsum
class DPSF(nn.Module):
    def __init__(self, d_model, d_state=64, expand=2, d_conv=4, conv_bias=True, bias=False):
        super().__init__()
        self.d_model = d_model  
        self.d_state = d_state  
        self.d_conv = d_conv  
        self.expand = expand  
        self.conv_bias = conv_bias
        self.bias = bias
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=self.bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=self.conv_bias,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
        )
        self.depth_conv2 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=5,
            stride=1,
            padding=5 // 2,
            groups=self.d_inner
        )
        self.point_conv2 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.DW = DWConv2(128, 128, kernel_size=5)
        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)

        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, self.d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=self.bias)

    def forward(self, x):
        """Mamba block forward.
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
        Returns:
            output: shape (b, l, d)
        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)
        x = rearrange(x, 'b l d_in -> b d_in l')
        x2=self.conv1d(x)[:, :, :l]
        x2 = rearrange(x2, 'b d_in l -> b l d_in')
        x2 = F.silu(x2)
        y = self.ssm(x2)
        xx1 = rearrange(res, 'b l d_in -> b d_in l')
        xx2 = self.conv1d(xx1)[:, :, :l]
        xx3= rearrange(xx2, 'b d_in l -> b l d_in')
        y = y * F.silu(xx3)
        output = self.out_proj(y)
        return output

    def ssm(self, x):
        """Runs the SSM.
        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)

        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)

        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        """
        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)

        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.

        """
        (b, l, d_in) = u.shape
        n = A.shape[1]

        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

        y = y + u * D

        return y

class DWConv(nn.Module):
    def __init__(self, dim=768,H_W=11):
        super(DWConv, self).__init__()
        self.H_W=H_W
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, self.H_W,self.H_W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class SRA(nn.Module):
    def __init__(self, dim : int, sr_ratio=2):
        super().__init__()
        #assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.q = nn.Conv2d(64, dim, kernel_size=1, groups=dim)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(64, dim, kernel_size=3,padding=1,groups=dim)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H*W
        x1 = x.reshape(B,N,C)
        if self.sr_ratio > 1:
            #x_ = x1.permute(0,2,1).reshape(B,C,H,W)
            x_ = self.sr(x)
            x_ = x_.reshape(B,-1,64)
            x2 = self.norm(x_)
        else:
            x2 = self.q(x).reshape(B,-1,64)
        return x2

class GLCA(nn.Module):

    def __init__(self, dim: int, head_dim: int, num_heads: int, num_patch: int, patch_size: int):
        super().__init__()
        self.dim = dim

        self.head_dim = head_dim
        self.num_patch = num_patch
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.inner_dim = head_dim * num_heads
        #self.inner_dim = head_dim
        self.scale = head_dim ** -0.5
        self.attn = nn.Softmax(dim=-1)
        self.osr = SRA(dim, num_heads)
#--------------------------------------------------------------------------------------
        self.act = nn.GELU()
        self.bn = nn.BatchNorm2d(self.dim)
        self.qkc = nn.Conv2d(self.dim, self.inner_dim * 3, kernel_size=1, padding=0, groups=head_dim, bias=False)
        self.spe = nn.Conv2d(dim, dim, kernel_size=1, padding=0, groups=head_dim, bias=False)
        self.bnc = nn.BatchNorm2d(self.inner_dim)
        self.bnc1 = nn.BatchNorm2d(dim)
        self.DW = DWConv1(dim, dim, kernel_size=5)
        self.avgpool=nn.AdaptiveAvgPool1d(dim)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        x = x.contiguous().view(b, self.dim, self.num_patch, self.num_patch)
        qy = self.osr(x)
        q = einops.rearrange(qy, "b n (h d) -> b h n d", h=self.num_heads)
        k = v = q
        scores = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        scores = scores * self.scale
        attn = self.attn(scores)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        oo = self.act(self.bnc1(self.DW(x)))
        oo = oo.reshape(b, self.dim, self.num_patch, self.num_patch).reshape(b, n, -1)
        out = self.avgpool(out+oo)
        return out

class DWConv1(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(DWConv1, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=out_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN = nn.BatchNorm2d(in_ch)

    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act1(out)
        # out = self.depth_conv(out)
        # out = self.Act2(out)
        return out

class DWConv2(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''

    def __init__(self, in_ch, out_ch, kernel_size=5):
        super(DWConv2, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=out_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN = nn.BatchNorm2d(in_ch)

    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act1(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        return out
class CIT(nn.Module):

    def __init__(self, dim: int, num_layers: int, num_heads: int, head_dim: int, hidden_dim: int, num_patch: int,
                 patch_size: int):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = [
                nn.Sequential(nn.LayerNorm(dim), GLCA(dim, head_dim, num_heads, num_patch, patch_size)),
                nn.Sequential(nn.LayerNorm(dim), DPSF(d_model=dim))
            ]
            self.layers.append(nn.ModuleList(layer))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x