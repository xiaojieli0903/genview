# Copyright (c) OpenMMLab. All rights reserved.
import io
import urllib
import numpy as np
import math
import os
from functools import reduce
from operator import mul
from typing import Dict, List, Optional, Union, Tuple
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from mmpretrain.models.backbones import VisionTransformer
from mmpretrain.models.utils import (build_2d_sincos_position_embedding,
                                     to_2tuple)
from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer)
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from ..utils import CosineEMA
from .base import BaseSelfSupervisor


@MODELS.register_module()
class MoCoV3(BaseSelfSupervisor):
    """MoCo v3.

    Implementation of `An Empirical Study of Training Self-Supervised Vision
    Transformers <https://arxiv.org/abs/2104.02057>`_.

    Args:
        backbone (dict): Config dict for module of backbone
        neck (dict): Config dict for module of deep features to compact feature
            vectors.
        head (dict): Config dict for module of head functions.
        base_momentum (float): Momentum coefficient for the momentum-updated
            encoder. Defaults to 0.01.
        pretrained (str, optional): The pretrained checkpoint path, support
            local path and remote path. Defaults to None.
        data_preprocessor (dict, optional): The config for preprocessing
            input data. If None or no specified type, it will use
            "SelfSupDataPreprocessor" as type.
            See :class:`SelfSupDataPreprocessor` for more details.
            Defaults to None.
        init_cfg (Union[List[dict], dict], optional): Config dict for weight
            initialization. Defaults to None.
        use_clip_mask (bool): Whether to use CLIP mask. Default: False.
        clip_model_name (str): CLIP model variant name. Default: 'convnext_base_w'.
        adjust_loss (bool): Whether to adjust loss computation. Default: True.
        gamma (int): Scaling factor for loss adjustment. Default: 1.
        distance_type (str): Type of distance metric for similarity. Default: 'cosine'.
        vis_input (bool): Whether to visualize inputs for debugging. Default: False.
    """

    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 base_momentum: float = 0.01,
                 pretrained: Optional[str] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[Union[List[dict], dict]] = None,
                 use_clip_mask: bool = False,
                 clip_model_name: str = 'convnext_base_w',
                 adjust_loss: bool = True,
                 gamma: int = 1,
                 distance_type: str = 'cosine',
                 vis_input: bool = False) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            pretrained=pretrained,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        # create momentum model
        if self.head.loss_module.loss_weight == 0:
            self.momentum_encoder = None
            self.head = None
        else:
            self.momentum_encoder = CosineEMA(
                nn.Sequential(self.backbone, self.neck),
                momentum=base_momentum)

        self.use_clip_mask = use_clip_mask
        self.clip_model_name = clip_model_name
        self.adjust_loss = adjust_loss
        self.gamma = gamma
        self.distance_type = distance_type
        self.vis_input = vis_input

        # Determine the mask type based on configuration
        if self.use_clip_mask:
            self.mask_name = 'clipmask_' + self.clip_model_name
        else:
            self.mask_name = ''

        # Load the CLIP model if the clip mask is used
        if self.use_clip_mask:
            import open_clip
            pretrained = 'laion2B-s13B-b82K-augreg'
            self.patch_size = 32
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.clip_model_name,
                pretrained=pretrained,
                device='cuda')
            self.clip_model = model
            self.clip_model.eval()
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.standard_array = torch.Tensor(
                np.load(
                    "tools/clip_pca/output_convnext_base_w_laion2B-s13B-b82K-augreg/pca_vectors.npy").reshape(
                    -1)).cuda()
            print(
                f'Load standard_array done, shape = {self.standard_array.shape}')
            self.patch_number = 224 // self.patch_size

    def make_foreground_softmask(self,
                                 tokens,
                                 grid_size: Tuple[int, int] = (16, 16),
                                 output_size: Tuple[int, int] = (16, 16),
                                 view_index: Optional[int] = None):
        """Generate foreground mask.

        Args:
            tokens (torch.Tensor): Input tensor of shape (bs, 16*16, 768).
            grid_size (Tuple[int, int], optional): Grid size of the input tensor. Default is (16, 16).
            output_size (Tuple[int, int], optional): Size of the output mask. Default is (16, 16).
            view_index (int, optional): Optional view index for visualization. Default is None.

        Returns:
            torch.Tensor: Output foreground mask of shape (bs, 1, output_size[0], output_size[1]).
        """
        # Reshape tokens to (bs * 16*16, 768)
        projection = (tokens.reshape(-1, tokens.shape[-1]
                                     ) @ self.standard_array.type_as(
            tokens)).reshape(-1, 1, *grid_size)

        map_fg = projection
        bs, channel = map_fg.shape[0], map_fg.shape[1]

        # Calculate min and max values for both map_fg and map_bg
        max_fg = map_fg.view(bs, channel, -1).max(dim=-1,
                                                  keepdim=True).values.view(
            bs, channel, 1, 1)
        min_fg = map_fg.view(bs, channel, -1).min(dim=-1,
                                                  keepdim=True).values.view(
            bs, channel, 1, 1)

        # Normalize map_fg and map_bg
        map_fg = (map_fg - min_fg) / (max_fg - min_fg + 1e-7)
        map_bg = 1 - map_fg

        if view_index is not None:
            for idx in range(map_fg.shape[0]):
                mask_resized = map_fg[idx].reshape(*output_size)
                mask_array = mask_resized.cpu().numpy()
                # Save visualized foreground masks
                mask_img = Image.fromarray(
                    (mask_array * 255).astype(np.uint8))
                img_name = f'examples/{self.mask_name}/{idx}_view{view_index}_fg_maps_{output_size[0]}x{output_size[1]}.jpeg'
                if not os.path.exists(os.path.dirname(img_name)):
                    print(f'Make dirs: {os.path.dirname(img_name)}')
                    os.makedirs(os.path.dirname(img_name))
                mask_img.save(img_name)
                mask_resized = map_bg[idx].reshape(*output_size)
                mask_array = mask_resized.cpu().numpy()
                # Save visualized foreground masks
                mask_img = Image.fromarray(
                    (mask_array * 255).astype(np.uint8))
                img_name = f'examples/{self.mask_name}/{idx}_view{view_index}_bg_maps_{output_size[0]}x{output_size[1]}.jpeg'
                mask_img.save(img_name)

        map_fg = map_fg / (
                    torch.sum(map_fg, dim=[2, 3]).view(bs, channel, 1,
                                                       1) + 1e-7)
        map_bg = map_bg / (
                    torch.sum(map_bg, dim=[2, 3]).view(bs, channel, 1,
                                                       1) + 1e-7)

        return [map_fg, map_bg]

    def calculate_weight(self, sim_foreground, sim_background):
        score = sim_foreground - sim_background
        score = torch.exp(self.gamma * score)
        loss_weight = score / score.sum() * score.shape[0]
        return score, loss_weight.detach()
    
    def aggregate_feature(self, masks, tokens):
        features = torch.einsum('bchw, bkhw->bk', masks,
                                tokens.permute(0, 2, 1).reshape(
                                    tokens.shape[0],
                                    tokens.shape[-1],
                                    self.patch_number,
                                    self.patch_number))
        return features

    def calculate_sim(self, feature_1, feature_2):
        if self.distance_type == 'cosine':
            sim_features = torch.sum(
                F.normalize(feature_1, dim=-1) * F.normalize(feature_2,
                                                             dim=-1),
                dim=-1)
        else:
            sim_features = torch.sum(feature_1 * feature_2, dim=-1)
        return sim_features
    
    def loss(self, inputs: List[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[DataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        assert isinstance(inputs, list)
        view_1 = inputs[0]
        view_2 = inputs[1]

        # compute query features, [N, C] each
        q1 = self.neck(self.backbone(view_1)[-1:])[0]
        q2 = self.neck(self.backbone(view_2)[-1:])[0]

        losses = {}
        loss_weight = None
        score = None

        # compute key features, [N, C] each, no gradient
        with torch.no_grad():
            # update momentum encoder
            self.momentum_encoder.update_parameters(
                nn.Sequential(self.backbone, self.neck))
            out1 = self.momentum_encoder(view_1)
            out2 = self.momentum_encoder(view_2)
            k1 = out1[-1]
            k2 = out2[-1]

        if self.use_clip_mask:
            with torch.no_grad(), torch.cuda.amp.autocast():
                tokens_1 = self.clip_model.encode_image(view_1)[1]
                tokens_2 = self.clip_model.encode_image(view_2)[1]

            masks_1 = self.make_foreground_softmask(
                tokens_1,
                grid_size=(self.patch_number, self.patch_number),
                output_size=(self.patch_number, self.patch_number),
                view_index=1 if self.vis_input else None
            )
            masks_2 = self.make_foreground_softmask(
                tokens_2,
                grid_size=(self.patch_number, self.patch_number),
                output_size=(self.patch_number, self.patch_number),
                view_index=2 if self.vis_input else None
            )

            z1_foreground = self.aggregate_feature(masks_1[0], tokens_1)
            z1_background = self.aggregate_feature(masks_1[1], tokens_1)
            z2_foreground = self.aggregate_feature(masks_2[0], tokens_2)
            z2_background = self.aggregate_feature(masks_2[1], tokens_2)
            
            sim_foreground = self.calculate_sim(z1_foreground, z2_foreground)
            sim_background = self.calculate_sim(z1_background, z2_background)

            if self.adjust_loss or self.vis_input:
                score, loss_weight = self.calculate_weight(sim_foreground, sim_background)
            if self.vis_input:
                std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                                   dtype=view_1.dtype).view(1, 3, 1,
                                                            1).type_as(
                    view_1)
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                                    dtype=view_1.dtype).view(1, 3, 1,
                                                             1).type_as(
                    view_1)
                for idx in range(view_1.shape[0]):
                    out_img = view_1[idx]
                    image_tensor = out_img * std + mean
                    modified_image_array = image_tensor.squeeze().permute(1, 2,
                                                                          0).cpu().numpy()
                    modified_image_array = (modified_image_array * 255).astype(
                        np.uint8)
                    combined_image = Image.fromarray(modified_image_array)
                    combined_image.save(f'examples/{self.mask_name}/{idx}_momentum_view1_{sim_foreground[idx]}_{sim_background[idx]}_{score[idx]}_{loss_weight[idx]}.jpeg')
                for idx in range(view_2.shape[0]):
                    out_img = view_2[idx]
                    image_tensor = out_img * std + mean
                    modified_image_array = image_tensor.squeeze().permute(1, 2,
                                                                          0).cpu().numpy()
                    modified_image_array = (modified_image_array * 255).astype(
                        np.uint8)
                    combined_image = Image.fromarray(modified_image_array)
                    combined_image.save(f'examples/{self.mask_name}/{idx}_momentum_view2_{sim_foreground[idx]}_{sim_background[idx]}_{score[idx]}_{loss_weight[idx]}.jpeg')

        loss = self.head.loss(
            q1, k2, data_samples, weight=loss_weight) + self.head.loss(
            q2, k1, data_samples, weight=loss_weight)

        losses['loss'] = loss
        return losses


@MODELS.register_module()
class MoCoV3ViT(VisionTransformer):
    """Vision Transformer for MoCoV3 pre-training.

    A pytorch implement of: `An Images is Worth 16x16 Words: Transformers for
    Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Part of the code is modified from:
    `<https://github.com/facebookresearch/moco-v3/blob/main/vits.py>`_.

    Args:
        stop_grad_conv1 (bool): whether to stop the gradient of
            convolution layer in `PatchEmbed`. Defaults to False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 stop_grad_conv1: bool = False,
                 frozen_stages: int = -1,
                 norm_eval: bool = False,
                 init_cfg: Optional[Union[dict, List[dict]]] = None,
                 **kwargs) -> None:

        # add MoCoV3 ViT-small arch
        self.arch_zoo.update(
            dict.fromkeys(
                ['mocov3-s', 'mocov3-small'], {
                    'embed_dims': 384,
                    'num_layers': 12,
                    'num_heads': 12,
                    'feedforward_channels': 1536,
                }))

        super().__init__(init_cfg=init_cfg, **kwargs)
        self.patch_size = kwargs['patch_size']
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.init_cfg = init_cfg

        if stop_grad_conv1:
            self.patch_embed.projection.weight.requires_grad = False
            self.patch_embed.projection.bias.requires_grad = False

        self._freeze_stages()

    def init_weights(self) -> None:
        """Initialize position embedding, patch embedding, qkv layers and cls
        token."""
        super().init_weights()

        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):

            # Use fixed 2D sin-cos position embedding
            pos_emb = build_2d_sincos_position_embedding(
                patches_resolution=self.patch_resolution,
                embed_dims=self.embed_dims,
                cls_token=True)
            self.pos_embed.data.copy_(pos_emb)
            self.pos_embed.requires_grad = False

            # xavier_uniform initialization for PatchEmbed
            val = math.sqrt(
                6. / float(3 * reduce(mul, to_2tuple(self.patch_size), 1) +
                           self.embed_dims))
            nn.init.uniform_(self.patch_embed.projection.weight, -val, val)
            nn.init.zeros_(self.patch_embed.projection.bias)

            # initialization for linear layers
            for name, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    if 'qkv' in name:
                        # treat the weights of Q, K, V separately
                        val = math.sqrt(
                            6. /
                            float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                        nn.init.uniform_(m.weight, -val, val)
                    else:
                        nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
            nn.init.normal_(self.cls_token, std=1e-6)

    def _freeze_stages(self) -> None:
        """Freeze patch_embed layer, some parameters and stages."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

            self.cls_token.requires_grad = False
            self.pos_embed.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = self.layers[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

            if i == (self.num_layers) and self.final_norm:
                for param in getattr(self, 'norm1').parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True) -> None:
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
