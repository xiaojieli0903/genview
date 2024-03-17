# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS
from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer)


@MODELS.register_module()
class MaskNeck(BaseModule):
    """The convolution mask neck.

    Args:
        in_channels (int): Number of input channels.
        hid_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        num_layers (int): Number of fc layers. Defaults to 2.
        with_bias (bool): Whether to use bias in fc layers (except for the
            last). Defaults to False.
        with_last_bn (bool): Whether to add the last BN layer.
            Defaults to True.
        with_last_bn_affine (bool): Whether to have learnable affine parameters
            in the last BN layer (set False for SimSiam). Defaults to True.
        with_last_bias (bool): Whether to use bias in the last fc layer.
            Defaults to False.
        with_avg_pool (bool): Whether to apply the global average pooling
            after backbone. Defaults to True.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to dict(type='SyncBN').
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        out_channels: int,
        num_layers: int = 2,
        with_bias: bool = False,
        with_last_bn: bool = True,
        with_last_bn_affine: bool = True,
        with_last_bias: bool = False,
        guide_foreground: dict = None,
        conv_cfg: dict = None,
        norm_cfg: dict = dict(type='SyncBN'),
        init_cfg: Optional[Union[dict, List[dict]]] = [
            dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
        ]
    ) -> None:
        super(MaskNeck, self).__init__(init_cfg)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.guide_foreground = guide_foreground
        self.relu = nn.ReLU(inplace=False)
        self.conv_bn_0 = ConvModule(
            in_channels,
            hid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            inplace=False)
        if self.guide_foreground is not None:
            self.guide_foreground_branch = nn.Linear(guide_foreground['in_channels'], guide_foreground['out_channels'])
        else:
            self.guide_foreground_branch = None
        self.conv_names = []
        self.bn_names = []
        for i in range(1, num_layers):
            this_channels = out_channels if i == num_layers - 1 \
                else hid_channels
            if i != num_layers - 1:
                self.add_module(
                    f'conv{i}',
                    nn.Conv2d(hid_channels, this_channels, kernel_size=3,
                              bias=with_bias))
                self.add_module(f'bn{i}',
                                build_norm_layer(norm_cfg, this_channels)[1])
                self.bn_names.append(f'bn{i}')
            else:
                self.add_module(
                    f'conv{i}',
                    nn.Conv2d(hid_channels, this_channels, kernel_size=1,
                              bias=with_bias))
                if with_last_bn:
                    self.add_module(
                        f'bn{i}',
                        build_norm_layer(
                            dict(**norm_cfg, affine=with_last_bn_affine),
                            this_channels)[1])
                    self.bn_names.append(f'bn{i}')
                else:
                    self.bn_names.append(None)
            self.conv_names.append(f'conv{i}')

    def forward(self, x: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        """Forward function.

        Args:
            x (Tuple[torch.Tensor]): The feature map of backbone.

        Returns:
            Tuple[torch.Tensor]: The output features.
        """
        assert len(x) == 1
        x = x[0]
        x = self.conv_bn_0(x)
        for conv_name, bn_name in zip(self.conv_names, self.bn_names):
            x = self.relu(x)
            conv = getattr(self, conv_name)
            x = conv(x)
            if bn_name is not None:
                bn = getattr(self, bn_name)
                x = bn(x)
        return (x, )
