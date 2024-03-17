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
import torch.nn.functional as F

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from .base import BaseSelfSupervisor


@MODELS.register_module()
class SimSiam(BaseSelfSupervisor):
    """SimSiam.

    Implementation of `Exploring Simple Siamese Representation Learning
    <https://arxiv.org/abs/2011.10566>`_. The operation of fixing learning rate
    of predictor is in `engine/hooks/simsiam_hook.py`.
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
        img_v1 = inputs[0]
        img_v2 = inputs[1]

        z1 = self.neck(self.backbone(img_v1))[0]  # NxC
        z2 = self.neck(self.backbone(img_v2))[0]  # NxC

        loss_weight = None
        score = None

        if self.use_clip_mask:
            with torch.no_grad(), torch.cuda.amp.autocast():
                tokens_1 = self.clip_model.encode_image(img_v1)[1]
                tokens_2 = self.clip_model.encode_image(img_v2)[1]
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
                score, loss_weight = self.calculate_weight(sim_foreground,
                                                           sim_background)
            if self.vis_input:
                std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                                   dtype=img_v1.dtype).view(1, 3, 1,
                                                            1).type_as(
                    img_v1)
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                                    dtype=img_v1.dtype).view(1, 3, 1,
                                                             1).type_as(
                    img_v1)
                for idx in range(img_v1.shape[0]):
                    out_img = img_v1[idx]
                    image_tensor = out_img * std + mean
                    modified_image_array = image_tensor.squeeze().permute(1, 2,
                                                                          0).cpu().numpy()
                    modified_image_array = (modified_image_array * 255).astype(
                        np.uint8)
                    combined_image = Image.fromarray(modified_image_array)
                    combined_image.save(f'examples/{self.mask_name}/{idx}_momentum_view1_{sim_foreground[idx]}_{sim_background[idx]}_{score[idx]}_{loss_weight[idx]}.jpeg')
                for idx in range(img_v2.shape[0]):
                    out_img = img_v2[idx]
                    image_tensor = out_img * std + mean
                    modified_image_array = image_tensor.squeeze().permute(1, 2,
                                                                          0).cpu().numpy()
                    modified_image_array = (modified_image_array * 255).astype(
                        np.uint8)
                    combined_image = Image.fromarray(modified_image_array)
                    combined_image.save(f'examples/{self.mask_name}/{idx}_momentum_view2_{sim_foreground[idx]}_{sim_background[idx]}_{score[idx]}_{loss_weight[idx]}.jpeg')

        loss_1 = self.head.loss(z1, z2, weight=loss_weight)
        loss_2 = self.head.loss(z2, z1, weight=loss_weight)

        losses = dict(loss=0.5 * (loss_1 + loss_2))
        return losses
