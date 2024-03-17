# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmengine.dist import all_gather, get_rank
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample


@MODELS.register_module()
class MoCoV3Head(BaseModule):
    """Head for MoCo v3 Pre-training.

    This head builds a predictor, which can be any registered neck component.
    It also implements latent contrastive loss between two forward features.
    Part of the code is modified from:
    `<https://github.com/facebookresearch/moco-v3/blob/main/moco/builder.py>`_.

    Args:
        predictor (dict): Config dict for module of predictor.
        loss (dict): Config dict for module of loss functions.
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Defaults to 1.0.
    """

    def __init__(self,
                 predictor: dict,
                 loss: dict,
                 temperature: float = 1.0) -> None:
        super().__init__()
        self.predictor = MODELS.build(predictor)
        self.loss_module = MODELS.build(loss)
        self.temperature = temperature
        self.loss_type = loss['type']

    def loss(self, base_out: torch.Tensor, momentum_out: torch.Tensor,
             data_samples: List[DataSample], weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate loss.

        Args:
            base_out (torch.Tensor): NxC features from base_encoder.
            momentum_out (torch.Tensor): NxC features from momentum_encoder.
            data_samples (List[DataSample]): All elements required
                during the forward function.
            weight: sample weight

        Returns:
            torch.Tensor: The loss tensor.
        """
        # predictor computation
        pred = self.predictor([base_out])[0]

        # normalize
        pred = nn.functional.normalize(pred, dim=1)
        target = nn.functional.normalize(momentum_out, dim=1)

        # get negative samples
        target = torch.cat(all_gather(target), dim=0)

        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [pred, target]) / self.temperature

        # generate labels
        batch_size = logits.shape[0]
        labels = (torch.arange(batch_size, dtype=torch.long) +
                  batch_size * get_rank()).to(logits.device)

        loss = self.loss_module(logits, labels, weight=weight)
        return loss


@MODELS.register_module()
class DiffMoCoV3Head(MoCoV3Head):
    """Head for MoCo v3 Pre-training.

    This head builds a predictor, which can be any registered neck component.
    It also implements latent contrastive loss between two forward features.
    Part of the code is modified from:
    `<https://github.com/facebookresearch/moco-v3/blob/main/moco/builder.py>`_.

    Args:
        predictor (dict): Config dict for module of predictor.
        loss (dict): Config dict for module of loss functions.
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Defaults to 1.0.
    """
    def __init__(self,
                 predictor: dict,
                 loss: dict,
                 temperature: float = 1.0,
                 adjust_temperature: bool = False,
                 map_temperature_scales: dict = {10: 1, 8: 2, 6: 3, 4: 4, 2: 5}
                 ) -> None:
        super().__init__(
            predictor=predictor,
            loss=loss,
            temperature=temperature
        )
        self.adjust_temperature = adjust_temperature
        self.map_temperature_scales = map_temperature_scales

    def loss(self, base_out: torch.Tensor,
             momentum_out: torch.Tensor,
             data_samples: List[DataSample],
             weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate loss.

        Args:
            base_out (torch.Tensor): NxC features from base_encoder.
            momentum_out (torch.Tensor): NxC features from momentum_encoder.
            data_samples (List[DataSample]): All elements required
                during the forward function.
        Returns:
            torch.Tensor: The loss tensor.
        """
        # predictor computation
        pred = self.predictor([base_out])[0]
        if self.loss_type == 'CrossEntropyLoss':
            # normalize
            pred = nn.functional.normalize(pred, dim=1)
            target = nn.functional.normalize(momentum_out, dim=1)

            # get negative samples
            target = torch.cat(all_gather(target), dim=0)

            if not self.adjust_temperature:
                # Einstein sum is more intuitive
                logits = torch.einsum(
                    'nc,mc->nm', [pred, target]) / self.temperature
            else:
                adjusted_temperatures = torch.Tensor([
                    self.map_temperature_scales[
                        data_sample.gen_img_scale] if hasattr(
                        data_sample, 'gen_img_scale'
                    ) else self.map_temperature_scales[10] for data_sample in
                    data_samples]).view(-1, 1).type_as(pred)
                logits = torch.einsum(
                    'nc,mc->nm', [pred, target]) / adjusted_temperatures

            # generate labels
            batch_size = logits.shape[0]
            labels = (torch.arange(batch_size, dtype=torch.long) +
                      batch_size * get_rank()).to(logits.device)

            loss = self.loss_module(logits, labels, weight=weight)
        else:
            # for MSE loss
            loss = self.loss_module(pred, momentum_out,)

        return loss
