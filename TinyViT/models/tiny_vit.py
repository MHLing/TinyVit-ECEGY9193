# --------------------------------------------------------
# TinyViT Model Architecture
# Copyright (c) 2022 Microsoft
# Adapted from LeViT and Swin Transformer
#   LeViT: (https://github.com/facebookresearch/levit)
#   Swin: (https://github.com/microsoft/swin-transformer)
# Build the TinyViT Model
# --------------------------------------------------------

import itertools
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import timm
from timm.models.layers import DropPath as TimmDropPath,\
    to_2tuple, trunc_normal_
from timm.models.registry import register_model
try:
    # timm.__version__ >= "0.6"
    from timm.models._builder import build_model_with_cfg
except (ImportError, ModuleNotFoundError):
    # timm.__version__ < "0.6"
    from timm.models.helpers import build_model_with_cfg


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class DropPath(TimmDropPath):
    def __init__(self, drop_prob=None):
        super().__init__(drop_prob=drop_prob)
        self.drop_prob = drop_prob

    def __repr__(self):
        msg = super().__repr__()
        msg += f'(drop_prob={self.drop_prob})'
        return msg


class PatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, resolution, activation):
        super().__init__()
        img_size: Tuple[int, int] = to_2tuple(resolution)
        self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
        self.num_patches = self.patches_resolution[0] * \
            self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        n = embed_dim
        self.seq = nn.Sequential(
            Conv2d_BN(in_chans, n // 2, 3, 2, 1),
            activation(),
            Conv2d_BN(n // 2, n, 3, 2, 1),
        )

    def forward(self, x):
        return self.seq(x)


class MBConv(nn.Module):
    def __init__(self, in_chans, out_chans, expand_ratio,
                 activation, drop_path):
        super().__init__()
        self.in_chans = in_chans
        self.hidden_chans = int(in_chans * expand_ratio)
        self.out_chans = out_chans

        self.conv1 = Conv2d_BN(in_chans, self.hidden_chans, ks=1)
        self.act1 = activation()

        self.conv2 = Conv2d_BN(self.hidden_chans, self.hidden_chans,
                               ks=3, stride=1, pad=1, groups=self.hidden_chans)
        self.act2 = activation()

        self.conv3 = Conv2d_BN(
            self.hidden_chans, out_chans, ks=1, bn_weight_init=0.0)
        self.act3 = activation()

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.conv3(x)

        x = self.drop_path(x)

        x += shortcut
        x = self.act3(x)

        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, out_dim, activation):
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.act = activation()
        self.conv1 = Conv2d_BN(dim, out_dim, 1, 1, 0)
        self.conv2 = Conv2d_BN(out_dim, out_dim, 3, 2, 1, groups=out_dim)
        self.conv3 = Conv2d_BN(out_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        if x.ndim == 3:
            H, W = self.input_resolution
            B, L, C = x.shape
            
            # Handle case where input has been pruned (L != H * W)
            if L != H * W:
                # Adjust H, W to match actual token count
                aspect_ratio = H / W if W > 0 else 1.0
                new_H = max(1, int((L * aspect_ratio) ** 0.5))
                new_W = L // new_H
                while new_H * new_W < L and new_H < L:
                    new_H += 1
                    new_W = L // new_H
                if new_H * new_W != L:
                    new_H = int(L ** 0.5)
                    new_W = (L + new_H - 1) // new_H
                H, W = new_H, new_W
            
            # (B, C, H, W)
            x = x.view(B, H, W, C).permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)

        x = x.flatten(2).transpose(1, 2)
        return x


class ConvLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth,
                 activation,
                 drop_path=0., downsample=None, use_checkpoint=False,
                 out_dim=None,
                 conv_expand_ratio=4.,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            MBConv(dim, dim, conv_expand_ratio, activation,
                   drop_path[i] if isinstance(drop_path, list) else drop_path,
                   )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, out_dim=out_dim, activation=activation)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=(14, 14),
                 return_attention=False,
                 ):
        super().__init__()
        # (h, w)
        assert isinstance(resolution, tuple) and len(resolution) == 2
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.return_attention = return_attention

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, h)
        self.proj = nn.Linear(self.dh, dim)

        points = list(itertools.product(
            range(resolution[0]), range(resolution[1])))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N),
                             persistent=False)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x, return_attention=None):  # x (B,N,C)
        B, N, _ = x.shape
        return_attention = return_attention if return_attention is not None else self.return_attention

        # Normalization
        x = self.norm(x)

        qkv = self.qkv(x)
        # (B, N, num_heads, d)
        q, k, v = qkv.view(B, N, self.num_heads, -
                           1).split([self.key_dim, self.key_dim, self.d], dim=3)
        # (B, num_heads, N, d)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (
            (q @ k.transpose(-2, -1)) * self.scale
            +
            (self.attention_biases[:, self.attention_bias_idxs]
             if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        
        if return_attention:
            return x, attn
        return x


class TinyViTBlock(nn.Module):
    r""" TinyViT Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int, int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        local_conv_size (int): the kernel size of the convolution between
                               Attention and MLP. Default: 3
        activation: the activation function. Default: nn.GELU
        token_pruning_ratio (float): Ratio of tokens to prune. Default: 0.0 (no pruning)
        token_pruning_method (str): Method for token pruning. Options: 'attention', 'magnitude'. Default: 'attention'
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 local_conv_size=3,
                 activation=nn.GELU,
                 token_pruning_ratio=0.0,
                 token_pruning_method='attention',
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        assert window_size > 0, 'window_size must be greater than 0'
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.token_pruning_ratio = token_pruning_ratio
        self.token_pruning_method = token_pruning_method

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        head_dim = dim // num_heads

        window_resolution = (window_size, window_size)
        self.attn = Attention(dim, head_dim, num_heads,
                              attn_ratio=1, resolution=window_resolution,
                              return_attention=(token_pruning_ratio > 0 and token_pruning_method == 'attention'))

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_activation = activation
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=mlp_activation, drop=drop)

        pad = local_conv_size // 2
        self.local_conv = Conv2d_BN(
            dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)
    
    def compute_token_importance(self, x, attn=None):
        """Compute token importance scores for pruning."""
        B, N, C = x.shape
        
        if self.token_pruning_method == 'attention' and attn is not None:
            # Use attention scores: importance = sum of attention received by each token
            # attn shape should be (B, num_heads, N, N) or compatible
            try:
                if attn.dim() == 4:
                    # Standard case: (B, num_heads, N, N)
                    attn_N = attn.shape[-1]  # Number of tokens in attention
                    if attn_N == N:
                        # For each token, sum the attention it receives from all other tokens
                        importance = attn.sum(dim=1).sum(dim=1)  # (B, N) - total attention received
                        # Normalize by number of tokens to get average
                        importance = importance / N
                    else:
                        # Shape mismatch, fallback to magnitude
                        importance = torch.norm(x, p=2, dim=-1)
                else:
                    # Unexpected attention shape, fallback to magnitude
                    importance = torch.norm(x, p=2, dim=-1)
            except Exception:
                # Any error, fallback to magnitude
                importance = torch.norm(x, p=2, dim=-1)
        elif self.token_pruning_method == 'magnitude':
            # Use feature magnitude: importance = L2 norm of token features
            importance = torch.norm(x, p=2, dim=-1)  # (B, N)
        else:
            # Default: use magnitude if attention not available
            importance = torch.norm(x, p=2, dim=-1)  # (B, N)
        
        # Ensure importance has correct shape
        if importance.shape != (B, N):
            # Fallback to magnitude if shape doesn't match
            importance = torch.norm(x, p=2, dim=-1)
        
        return importance
    
    def prune_tokens(self, x, importance, keep_indices=None):
        """Prune tokens based on importance scores."""
        if self.token_pruning_ratio <= 0 or keep_indices is not None:
            if keep_indices is not None:
                # Use provided indices
                B = x.shape[0]
                pruned_x = torch.gather(x, 1, keep_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
                return pruned_x, keep_indices
            return x, None
        
        B, N, C = x.shape
        
        # Ensure importance has the correct shape
        if importance.shape != (B, N):
            # If importance shape doesn't match, try to fix it
            if importance.dim() == 2 and importance.shape[0] == B:
                # Take first N elements if importance has more tokens
                if importance.shape[1] > N:
                    importance = importance[:, :N]
                # Pad if importance has fewer tokens
                elif importance.shape[1] < N:
                    padding = torch.zeros(B, N - importance.shape[1], device=importance.device, dtype=importance.dtype)
                    importance = torch.cat([importance, padding], dim=1)
            else:
                # Fallback: use magnitude
                importance = torch.norm(x, p=2, dim=-1)
        
        num_tokens_to_keep = max(1, int(N * (1 - self.token_pruning_ratio)))
        # Ensure we don't try to keep more tokens than available
        num_tokens_to_keep = min(num_tokens_to_keep, N)
        
        # Get top-k important tokens
        _, top_indices = torch.topk(importance, num_tokens_to_keep, dim=1)  # (B, num_tokens_to_keep)
        top_indices, _ = torch.sort(top_indices, dim=1)  # Sort to maintain spatial order
        
        # Gather pruned tokens
        pruned_x = torch.gather(x, 1, top_indices.unsqueeze(-1).expand(-1, -1, C))
        
        return pruned_x, top_indices

    def forward(self, x, keep_indices=None):
        H, W = self.input_resolution
        B, L, C = x.shape
        
        # Handle case where input has been pruned (L != H * W)
        if L != H * W:
            # Adjust H, W to match actual token count
            # Try to maintain aspect ratio
            aspect_ratio = H / W if W > 0 else 1.0
            new_H = max(1, int((L * aspect_ratio) ** 0.5))
            new_W = L // new_H
            # Ensure new_H * new_W == L
            while new_H * new_W < L and new_H < L:
                new_H += 1
                new_W = L // new_H
            if new_H * new_W != L:
                # Fallback: use square-like shape
                new_H = int(L ** 0.5)
                new_W = (L + new_H - 1) // new_H
            H, W = new_H, new_W
        
        res_x = x
        
        # Attention forward pass
        if H == self.window_size and W == self.window_size:
            # Non-windowed attention (exact match)
            if self.token_pruning_ratio > 0 and self.token_pruning_method == 'attention':
                x, attn = self.attn(x, return_attention=True)
            else:
                x = self.attn(x)
                attn = None
        else:
            # Windowed attention
            x = x.view(B, H, W, C)
            pad_b = (self.window_size - H %
                     self.window_size) % self.window_size
            pad_r = (self.window_size - W %
                     self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size
            # window partition
            x = x.view(B, nH, self.window_size, nW, self.window_size, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_size * self.window_size, C
            )
            attn_result = self.attn(x)
            # Handle both tuple (x, attn) and single tensor return
            if isinstance(attn_result, tuple):
                x, attn = attn_result
            else:
                x = attn_result
                attn = None  # Window attention doesn't easily support attention-based pruning
            
            # window reverse
            x = x.view(B, nH, nW, self.window_size, self.window_size,
                       C).transpose(2, 3).reshape(B, pH, pW, C)

            if padding:
                x = x[:, :H, :W].contiguous()

            x = x.view(B, H * W, C)
            L = H * W

        # Residual connection
        x = res_x + self.drop_path(x)

        # Token pruning after attention (works for both attention and magnitude methods)
        if self.token_pruning_ratio > 0:
            # Compute importance
            if self.token_pruning_method == 'attention' and attn is not None:
                importance = self.compute_token_importance(x, attn)
            else:
                # Use magnitude method (fallback for windowed attention or when attention not available)
                importance = self.compute_token_importance(x)
            
            # Prune tokens
            x, keep_indices = self.prune_tokens(x, importance, keep_indices)
            L = x.shape[1]
            
            # Update spatial dimensions for local conv
            aspect_ratio = H / W if W > 0 else 1.0
            new_H = max(1, int((L * aspect_ratio) ** 0.5))
            new_W = L // new_H
            while new_H * new_W < L and new_H < L:
                new_H += 1
                new_W = L // new_H
            if new_H * new_W != L:
                new_H = int(L ** 0.5)
                new_W = (L + new_H - 1) // new_H
            H, W = new_H, new_W
        
        # Reshape for local conv
        if L == H * W:
            x = x.transpose(1, 2).reshape(B, C, H, W)
            x = self.local_conv(x)
            x = x.view(B, C, L).transpose(1, 2)
        else:
            # Handle dimension mismatch - pad or truncate to match H*W
            target_L = H * W
            if L < target_L:
                padding = torch.zeros(B, target_L - L, C, device=x.device, dtype=x.dtype)
                x = torch.cat([x, padding], dim=1)
            elif L > target_L:
                x = x[:, :target_L, :]
            x = x.transpose(1, 2).reshape(B, C, H, W)
            x = self.local_conv(x)
            x = x.view(B, C, H * W).transpose(1, 2)
            L = H * W

        x = x + self.drop_path(self.mlp(x))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"


class BasicLayer(nn.Module):
    """ A basic TinyViT layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        local_conv_size: the kernel size of the depthwise convolution between attention and MLP. Default: 3
        activation: the activation function. Default: nn.GELU
        out_dim: the output dimension of the layer. Default: dim
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., drop=0.,
                 drop_path=0., downsample=None, use_checkpoint=False,
                 local_conv_size=3,
                 activation=nn.GELU,
                 out_dim=None,
                 token_pruning_ratio=0.0,
                 token_pruning_method='attention',
                 prune_first_block_only=False,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            # 如果只在第一个 block prune，那么 i==0 用 ratio，其它 i>0 强制 0
            if prune_first_block_only and i > 0:
                block_prune_ratio = 0.0
            else:
                block_prune_ratio = token_pruning_ratio

            blk = TinyViTBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                local_conv_size=local_conv_size,
                activation=activation,
                token_pruning_ratio=block_prune_ratio,   # ★ 每个 block 自己的 ratio
                token_pruning_method=token_pruning_method,
            )
            self.blocks.append(blk)

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, out_dim=out_dim, activation=activation)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class TinyViT(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000,
                 embed_dims=[96, 192, 384, 768], depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_sizes=[7, 7, 14, 7],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 use_checkpoint=False,
                 mbconv_expand_ratio=4.0,
                 local_conv_size=3,
                 layer_lr_decay=1.0,
                 token_pruning_ratio=0.0,
                 token_pruning_method='attention',
                 ):
        super().__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio

        activation = nn.GELU

        self.patch_embed = PatchEmbed(in_chans=in_chans,
                                      embed_dim=embed_dims[0],
                                      resolution=img_size,
                                      activation=activation)

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            kwargs = dict(dim=embed_dims[i_layer],
                          input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                            patches_resolution[1] // (2 ** i_layer)),
                          depth=depths[i_layer],
                          drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                          downsample=PatchMerging if (
                              i_layer < self.num_layers - 1) else None,
                          use_checkpoint=use_checkpoint,
                          out_dim=embed_dims[min(
                              i_layer + 1, len(embed_dims) - 1)],
                          activation=activation,
                          )
            if i_layer == 0:
                layer = ConvLayer(
                    conv_expand_ratio=mbconv_expand_ratio,
                    **kwargs,
                )
            else:
                # Only enable token pruning in Stage 3 (i_layer == 2).
                # Other stages get token_pruning_ratio = 0.0 to avoid
                # breaking window attention or pruning when token count is small.
                if i_layer == 2:
                    stage_token_pruning_ratio = token_pruning_ratio
                    prune_first_block_only = True
                else:
                    # Stage2 & Stage4：完全不 prune
                    stage_token_pruning_ratio = 0.0
                    prune_first_block_only = False
                layer = BasicLayer(
                    num_heads=num_heads[i_layer],
                    window_size=window_sizes[i_layer],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    token_pruning_ratio=stage_token_pruning_ratio,
                    token_pruning_method=token_pruning_method,
                    prune_first_block_only=prune_first_block_only,
                    **kwargs)
            self.layers.append(layer)

        # Classifier head
        self.norm_head = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(
            embed_dims[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

        # init weights
        self.apply(self._init_weights)
        self.set_layer_lr_decay(layer_lr_decay)

    def set_layer_lr_decay(self, layer_lr_decay):
        decay_rate = layer_lr_decay

        # layers -> blocks (depth)
        depth = sum(self.depths)
        lr_scales = [decay_rate ** (depth - i - 1) for i in range(depth)]

        def _set_lr_scale(m, scale):
            for p in m.parameters():
                p.lr_scale = scale

        self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))
        i = 0
        for layer in self.layers:
            for block in layer.blocks:
                block.apply(lambda x: _set_lr_scale(x, lr_scales[i]))
                i += 1
            if layer.downsample is not None:
                layer.downsample.apply(
                    lambda x: _set_lr_scale(x, lr_scales[i - 1]))
        assert i == depth
        for m in [self.norm_head, self.head]:
            m.apply(lambda x: _set_lr_scale(x, lr_scales[-1]))

        for k, p in self.named_parameters():
            p.param_name = k

        def _check_lr_scale(m):
            for p in m.parameters():
                assert hasattr(p, 'lr_scale'), p.param_name

        self.apply(_check_lr_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'attention_biases'}

    def forward_features(self, x):
        # x: (N, C, H, W)
        x = self.patch_embed(x)

        x = self.layers[0](x)
        start_i = 1

        for i in range(start_i, len(self.layers)):
            layer = self.layers[i]
            x = layer(x)

        x = x.mean(1)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.norm_head(x)
        x = self.head(x)
        return x


_checkpoint_url_format = \
    'https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/{}.pth'


def _create_tiny_vit(variant, pretrained=False, **kwargs):
    # pretrained_type: 22kto1k_distill, 1k, 22k_distill
    pretrained_type = kwargs.pop('pretrained_type', '22kto1k_distill')
    assert pretrained_type in ['22kto1k_distill', '1k', '22k_distill'], \
        'pretrained_type should be one of 22kto1k_distill, 1k, 22k_distill'

    img_size = kwargs.get('img_size', 224)
    if img_size != 224:
        pretrained_type = pretrained_type.replace('_', f'_{img_size}_')

    num_classes_pretrained = 21841 if \
        pretrained_type  == '22k_distill' else 1000

    variant_without_img_size = '_'.join(variant.split('_')[:-1])
    cfg = dict(
        url=_checkpoint_url_format.format(
            f'{variant_without_img_size}_{pretrained_type}'),
        num_classes=num_classes_pretrained,
        classifier='head',
    )

    def _pretrained_filter_fn(state_dict):
        state_dict = state_dict['model']
        # filter out attention_bias_idxs
        state_dict = {k: v for k, v in state_dict.items() if \
            not k.endswith('attention_bias_idxs')}
        return state_dict

    if timm.__version__ >= "0.6":
        return build_model_with_cfg(
            TinyViT, variant, pretrained,
            pretrained_cfg=cfg,
            pretrained_filter_fn=_pretrained_filter_fn,
            **kwargs)
    else:
        return build_model_with_cfg(
            TinyViT, variant, pretrained,
            default_cfg=cfg,
            pretrained_filter_fn=_pretrained_filter_fn,
            **kwargs)


@register_model
def tiny_vit_5m_224(pretrained=False, **kwargs):
    model_kwargs = dict(
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=0.0,
    )
    model_kwargs.update(kwargs)
    return _create_tiny_vit('tiny_vit_5m_224', pretrained, **model_kwargs)


@register_model
def tiny_vit_11m_224(pretrained=False, **kwargs):
    model_kwargs = dict(
        embed_dims=[64, 128, 256, 448],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 8, 14],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=0.1,
    )
    model_kwargs.update(kwargs)
    return _create_tiny_vit('tiny_vit_11m_224', pretrained, **model_kwargs)


@register_model
def tiny_vit_21m_224(pretrained=False, **kwargs):
    model_kwargs = dict(
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=0.2,
    )
    model_kwargs.update(kwargs)
    return _create_tiny_vit('tiny_vit_21m_224', pretrained, **model_kwargs)


@register_model
def tiny_vit_21m_384(pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=384,
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[12, 12, 24, 12],
        drop_path_rate=0.1,
    )
    model_kwargs.update(kwargs)
    return _create_tiny_vit('tiny_vit_21m_384', pretrained, **model_kwargs)


@register_model
def tiny_vit_21m_512(pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=512,
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[16, 16, 32, 16],
        drop_path_rate=0.1,
    )
    model_kwargs.update(kwargs)
    return _create_tiny_vit('tiny_vit_21m_512', pretrained, **model_kwargs)
