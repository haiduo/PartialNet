# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import timm
from timm.models.layers import DropPath, trunc_normal_
from functools import partial
from typing import List
from torch import Tensor
import torch.nn.functional as F
import copy
import os
from .dgconv import DGConv2d
from .irpe import build_rpe, get_rpe_config

try:
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint
    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first")
    has_mmdet = False


# fixed_list=[3,8,8,4,32,8,4,8,8,16,64,256,256]
# div_index = 0

def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 4
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None, act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x  
    
class RPEAttention(nn.Module):
    '''Attention with image relative position encoding'''
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, rpe_config=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # image relative position encoding
        self.rpe_q, self.rpe_k, self.rpe_v = build_rpe(rpe_config, head_dim=head_dim, num_heads=num_heads)

    def forward(self, x):
        B, C, h, w = x.shape
        x = x.view(B, C, h*w).transpose(1,2)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q *= self.scale

        attn = (q @ k.transpose(-2, -1))

        # image relative position on keys
        if self.rpe_k is not None:
            #attn += self.rpe_k(q)
            attn += self.rpe_k(q, h, w)
        # image relative position on queries
        if self.rpe_q is not None:
            attn += self.rpe_q(k * self.scale).transpose(2, 3)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v

        # image relative position on values
        if self.rpe_v is not None:
            out += self.rpe_v(attn)

        x = out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.transpose(1,2).view(B, C, h, w)
        return x

class SRM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.cfc1 = nn.Conv2d(channel, channel, kernel_size=(1,2), bias=False)
        #self.cfc2 = nn.Conv2d(channel, channel, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channel)
        self.sigmoid = nn.Hardsigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        # style pooling
        mean = x.reshape(b, c, -1).mean(-1).view(b,c,1,1)
        std = x.reshape(b, c, -1).std(-1).view(b,c,1,1)
        #max_value = torch.max(x.reshape(b, c, -1), -1)[0].view(b,c,1,1)
        u = torch.cat([mean, std], dim=-1)
        # style integration
        z = self.cfc1(u)
        #z = self.act(z)
        #z = self.cfc2(z)
        z = self.bn(z)
        g = self.sigmoid(z)
        g = g.reshape(b, c, 1, 1)
        return x * g.expand_as(x)

class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div, forward_type, use_attn='', channel_type='', patnet_t0=False):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim = dim
        self.n_div = n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.use_attn = use_attn
        self.channel_type = channel_type

        if use_attn:
            if channel_type == 'self':
                self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
                rpe_config = get_rpe_config(
                        ratio=20,
                        method="euc",
                        mode='bias',
                        shared_head=False,
                        skip=0,
                        rpe_on='k',
                    )
                if patnet_t0:
                    num_heads = 4
                else:
                    num_heads = 6
                self.attn = RPEAttention(self.dim_untouched, num_heads=num_heads, attn_drop=0.1, proj_drop=0.1, rpe_config=rpe_config)
                self.norm = timm.layers.LayerNorm2d(self.dim_untouched)
                #self.norm = timm.layers.LayerNorm2d(self.dim)
                self.forward = self.forward_atten
            elif channel_type == 'se':
                self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
                self.attn = SRM(self.dim_untouched)
                self.norm = nn.BatchNorm2d(self.dim_untouched)
                self.forward = self.forward_atten
        else:
            if forward_type == 'slicing':
                self.forward = self.forward_slicing
            elif forward_type == 'split_cat':
                self.forward = self.forward_split_cat
            else:
                raise NotImplementedError
    
    def forward_atten(self, x: Tensor) -> Tensor:
        if self.channel_type:
            # print(self.channel_type)
            if self.channel_type == 'se':
                x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
                x1 = self.partial_conv3(x1)
                #x = self.partial_conv3(x)
                x2 = self.attn(x2)
                x2 = self.norm(x2)
                x = torch.cat((x1, x2), 1)
                #x = self.attn(x)
            else:
                x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
                x1 = self.partial_conv3(x1)
                x2 = self.norm(x2)
                x2 = self.attn(x2)
                x = torch.cat((x1, x2), 1)
        return x
    
    def forward_slicing(self, x: Tensor) -> Tensor:
        x1 = x.clone()   # !!! Keep the original input intact for the residual connection later
        x1[:, :self.dim_conv3, :, :] = self.partial_conv3(x1[:, :self.dim_conv3, :, :])
        return x1

    def forward_split_cat(self, x: Tensor) -> Tensor:
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class partial_spatial_attn_layer_reverse(nn.Module):
    def __init__(self, dim, n_head, partial=0.5):
        super().__init__()
        self.dim = dim
        self.dim_conv = int(partial * dim)
        self.dim_untouched = dim - self.dim_conv
        self.nhead = n_head
        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, 1, bias=False)
        self.conv_attn = nn.Conv2d(self.dim_untouched, n_head, 1, bias=False)
        self.norm = nn.BatchNorm2d(self.dim_untouched)
        self.norm2 = nn.BatchNorm2d(self.dim_conv)
        #self.act2 = nn.GELU()
        self.act = nn.Hardsigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        x1, x2 = torch.split(x, [self.dim_untouched, self.dim_conv], 1)
        weight =self.act(self.conv_attn(x1))
        x1 = x1 * weight
        x1 = self.norm(x1)
        #x2 = self.act2(x2)
        x2 = self.norm2(x2)
        x2 = self.conv(x2)
        x = torch.cat((x1, x2), 1)
        return x


class MLPBlock(nn.Module):
    def __init__(self,
                 dim,
                 n_div,
                 print_n_div,
                 auto_div,
                 u_regular,
                 l_gate,
                 index_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 act_layer,
                 norm_layer,
                 pconv_fw_type,
                 pre_epoch,
                 use_channel,
                 use_spatial,
                 channel_type,
                 patnet_t0 = True
                 ):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity() 
        self.n_div = n_div
        self.print_n_div=print_n_div
        self.auto_div = auto_div
        self.u_regular = u_regular
        self.l_gate = l_gate
        self.index_div=index_div
        self.pre_epoch=pre_epoch
        self.split_shortcut = True if channel_type == "self" else False

        if self.auto_div:
            self.spatial_mixing = DGConv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, sort=False, 
                        u_regular=self.u_regular, l_gate=self.l_gate, index_div=self.index_div, n_div=self.n_div, 
                        print_n_div=self.print_n_div, pre_epoch=self.pre_epoch )
        else:
            self.spatial_mixing = Partial_conv3(
                dim, 
                n_div, 
                pconv_fw_type, 
                use_channel,
                channel_type,
                patnet_t0,
                )
            
        mlp_hidden_dim = int(dim * mlp_ratio)
        if use_spatial:
            mlp_layer: List[nn.Module] = [
                nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
                norm_layer(mlp_hidden_dim),
                act_layer(),
                nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False),
                partial_spatial_attn_layer_reverse(dim, 1)]
        else:
            mlp_layer: List[nn.Module] = [
                nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),  
                norm_layer(mlp_hidden_dim),
                act_layer(),
                nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)]

        self.mlp = nn.Sequential(*mlp_layer)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x: Tensor) -> Tensor:
        if self.split_shortcut:
            #x = self.layer_norm(x + self.spatial_mixing(x))
            x = x + self.spatial_mixing(x)
            x = x + self.drop_path(self.mlp(x))
        else:
            shortcut = x 
            x = self.spatial_mixing(x)
            x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x: Tensor) -> Tensor:
        if self.split_shortcut:
            x = x + self.spatial_mixing(x)
            x = x + self.drop_path(self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        else:
            shortcut = x
            x = self.spatial_mixing(x)
            x = shortcut + self.drop_path(self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x

class BasicStage(nn.Module):
    def __init__(self,
                dim,
                depth,
                n_div,
                print_n_div,
                auto_div,
                u_regular,
                l_gate,
                index_div,
                mlp_ratio,
                drop_path,
                layer_scale_init_value,
                norm_layer,
                act_layer,
                pconv_fw_type,
                pre_epoch,
                use_channel,
                use_spatial,
                channel_type='',
                patnet_t0 =True
                ):
        super().__init__()

        blocks_list = [
            MLPBlock(
                dim=dim,
                n_div=n_div,
                print_n_div=print_n_div,
                auto_div=auto_div,
                u_regular=u_regular,
                l_gate=l_gate,
                index_div=index_div,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                pconv_fw_type=pconv_fw_type,
                pre_epoch=pre_epoch,
                use_channel=use_channel,
                use_spatial=use_spatial,
                channel_type=channel_type,
                patnet_t0= patnet_t0,
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, patch_stride, in_chans, embed_dim, norm_layer):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.proj(x))
        return x


class PatchMerging(nn.Module):
    def __init__(self, patch_size2, patch_stride2, dim, norm_layer):
        super().__init__()
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=patch_size2, stride=patch_stride2, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(2 * dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.reduction(x))
        return x


class PartialNet(nn.Module):
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=96,
                 depths=(1, 2, 8, 2),
                 mlp_ratio=2.,
                 n_div=4,
                 print_n_div=False,
                 auto_div=False,
                 u_regular=False,
                 l_gate=False,
                 index_div=False,
                 patch_size=4,
                 patch_stride=4,
                 patch_size2=2,  # for subsequent layers
                 patch_stride2=2,
                 patch_norm=True,
                 feature_dim=1280,
                 drop_path_rate=0.1,
                 layer_scale_init_value=0,
                 norm_layer='BN',
                 act_layer='RELU',
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 pconv_fw_type='split_cat',
                 pre_epoch=100,
                 use_channel_attn=True,
                 use_spatial_attn=True,
                 patnet_t0=True,
                 **kwargs):
        super().__init__()

        if norm_layer == 'BN':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError

        if act_layer == 'GELU':
            act_layer = nn.GELU
        elif act_layer == 'RELU':
            act_layer = partial(nn.ReLU, inplace=True)
        else:
            raise NotImplementedError

        if not fork_feat:
            self.num_classes = num_classes
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_stages - 1))
        self.mlp_ratio = mlp_ratio
        self.depths = depths

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            patch_stride=patch_stride,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        stages_list = []
        for i_stage in range(self.num_stages):
            stage = BasicStage(dim=int(embed_dim * 2 ** i_stage),
                               n_div=n_div,
                               print_n_div=print_n_div,
                               auto_div=auto_div,
                               u_regular=u_regular,
                               l_gate=l_gate,
                               index_div=index_div,
                               depth=depths[i_stage],
                               mlp_ratio=self.mlp_ratio,
                               drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                               layer_scale_init_value=layer_scale_init_value,
                               norm_layer=norm_layer,
                               act_layer=act_layer,
                               pconv_fw_type=pconv_fw_type,
                               pre_epoch=pre_epoch,
                               use_channel=use_channel_attn,
                               use_spatial=use_spatial_attn,
                               channel_type='se' if i_stage<=2 else 'self', #'se'
                               patnet_t0=patnet_t0,
                               )
            stages_list.append(stage)

            # patch merging layer
            if i_stage < self.num_stages - 1:
                stages_list.append(PatchMerging(
                                        patch_size2=patch_size2,
                                        patch_stride2=patch_stride2,
                                        dim=int(embed_dim * 2 ** i_stage),
                                        norm_layer=norm_layer))

        self.stages = nn.Sequential(*stages_list)

        self.fork_feat = fork_feat

        if self.fork_feat:
            self.forward = self.forward_det
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    raise NotImplementedError
                else:
                    layer = norm_layer(int(embed_dim * 2 ** i_emb))
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            self.forward = self.forward_cls
            # Classifier head
            self.avgpool_pre_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.num_features, feature_dim, 1, bias=False),
                act_layer()
            )
            self.head = nn.Linear(feature_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self.cls_init_weights)
        self.init_cfg = copy.deepcopy(init_cfg)
        if self.fork_feat and (self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # init for mmdetection by loading imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)

            # show for debug
            print('missing_keys: ', missing_keys)
            print('unexpected_keys: ', unexpected_keys)

    def forward_cls(self, x):
        # output only the features of last layer for image classification
        x = self.patch_embed(x)
        x = self.stages(x)
        x = self.avgpool_pre_head(x)  # B C 1 1
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x

    def forward_det(self, x: Tensor) -> Tensor:
        # output the features of four stages for dense prediction
        x = self.patch_embed(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)

        return outs