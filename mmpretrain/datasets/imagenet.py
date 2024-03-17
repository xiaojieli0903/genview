# Copyright (c) OpenMMLab. All rights reserved.
import os
import random
import torch
from typing import Any, List, Optional, Union

from mmengine import fileio
from mmengine.logging import MMLogger

from mmpretrain.registry import DATASETS
from .categories import IMAGENET_CATEGORIES
from .custom import CustomDataset
from torch.utils.data import Dataset
from PIL import Image

from mmengine.fileio import (BaseStorageBackend, get_file_backend,
                             list_from_file, dict_from_file)


@DATASETS.register_module()
class ImageNet(CustomDataset):
    """`ImageNet <http://www.image-net.org>`_ Dataset.

    The dataset supports two kinds of directory format,

    ::

        imagenet
        ├── train
        │   ├──class_x
        |   |   ├── x1.jpg
        |   |   ├── x2.jpg
        |   |   └── ...
        │   ├── class_y
        |   |   ├── y1.jpg
        |   |   ├── y2.jpg
        |   |   └── ...
        |   └── ...
        ├── val
        │   ├──class_x
        |   |   └── ...
        │   ├── class_y
        |   |   └── ...
        |   └── ...
        └── test
            ├── test1.jpg
            ├── test2.jpg
            └── ...

    or ::

        imagenet
        ├── train
        │   ├── x1.jpg
        │   ├── y1.jpg
        │   └── ...
        ├── val
        │   ├── x3.jpg
        │   ├── y3.jpg
        │   └── ...
        ├── test
        │   ├── test1.jpg
        │   ├── test2.jpg
        │   └── ...
        └── meta
            ├── train.txt
            └── val.txt


    Args:
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        split (str): The dataset split, supports "train", "val" and "test".
            Default to ''.
        data_prefix (str | dict): Prefix for training data. Defaults to ''.
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        **kwargs: Other keyword arguments in :class:`CustomDataset` and
            :class:`BaseDataset`.


    Examples:
        >>> from mmpretrain.datasets import ImageNet
        >>> train_dataset = ImageNet(data_root='data/imagenet', split='train')
        >>> train_dataset
        Dataset ImageNet
            Number of samples:  1281167
            Number of categories:       1000
            Root of dataset:    data/imagenet
        >>> test_dataset = ImageNet(data_root='data/imagenet', split='val')
        >>> test_dataset
        Dataset ImageNet
            Number of samples:  50000
            Number of categories:       1000
            Root of dataset:    data/imagenet
    """  # noqa: E501

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    METAINFO = {'classes': IMAGENET_CATEGORIES}

    def __init__(self,
                 data_root: str = '',
                 split: str = '',
                 data_prefix: Union[str, dict] = '',
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 **kwargs):
        kwargs = {'extensions': self.IMG_EXTENSIONS, **kwargs}

        if split:
            splits = ['train', 'val', 'test']
            assert split in splits, \
                f"The split must be one of {splits}, but get '{split}'"

            if split == 'test':
                logger = MMLogger.get_current_instance()
                logger.info(
                    'Since the ImageNet1k test set does not provide label'
                    'annotations, `with_label` is set to False')
                kwargs['with_label'] = False

            data_prefix = split if data_prefix == '' else data_prefix

            if ann_file == '':
                _ann_path = fileio.join_path(data_root, 'meta', f'{split}.txt')
                if fileio.exists(_ann_path):
                    ann_file = fileio.join_path('meta', f'{split}.txt')

        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            ann_file=ann_file,
            metainfo=metainfo,
            **kwargs)

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Root of dataset: \t{self.data_root}',
        ]
        return body


@DATASETS.register_module()
class ImageNetDiffusion(ImageNet):
    """Dataset class for ImageNet Diffusion.

    This dataset is an extension of the ImageNet dataset with added support
    for a generated image dataset. It loads image paths and optional generated
    image paths.

    The dataset supports two kinds of directory format,

    ::

        imagenet
        ├── train
        │   ├──class_x
        |   |   ├── x1.jpg
        |   |   ├── x2.jpg
        |   |   └── ...
        │   ├── class_y
        |   |   ├── y1.jpg
        |   |   ├── y2.jpg
        |   |   └── ...
        |   └── ...
        ├── val
        │   ├──class_x
        |   |   └── ...
        │   ├── class_y
        |   |   └── ...
        |   └── ...
        └── test
            ├── test1.jpg
            ├── test2.jpg
            └── ...

    or ::

        imagenet
        ├── train
        │   ├── x1.jpg
        │   ├── y1.jpg
        │   └── ...
        ├── val
        │   ├── x3.jpg
        │   ├── y3.jpg
        │   └── ...
        ├── test
        │   ├── test1.jpg
        │   ├── test2.jpg
        │   └── ...
        └── meta
            ├── train.txt
            └── val.txt


   Args:
        data_root (str): The root directory for the main dataset.
        gen_data_root (str): The root directory for the generated dataset.
        gen_data_file (str): Path to a file containing generated image paths.
        split (str): The dataset split, supports "train", "val", and "test".
        data_prefix (str | dict): Prefix for training data.
        ann_file (str): Annotation file path.
        metainfo (dict, optional): Meta information for the dataset.
        **kwargs: Other keyword arguments for CustomDataset and BaseDataset.

    Examples:
        >>> from mmpretrain.datasets import ImageNetDiffusion
        >>> train_dataset = ImageNetDiffusion(data_root='data/imagenet', 
                                              gen_data_root='data/gen_imagenet',
                                              gen_data_file='list.txt',
                                              split='train')
        >>> train_dataset
        Dataset ImageNetDiffusion
            Number of samples:  1281167
            Number of categories:  1000
            Number of generated images:  50000
            Root of main dataset:  data/imagenet
            Root of generated dataset:  data/gen_imagenet
    """  # noqa: E501

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    METAINFO = {'classes': IMAGENET_CATEGORIES}

    def __init__(self,
                 data_root: str = '',
                 gen_data_root: str = '',
                 gen_data_file: str = '',
                 split: str = '',
                 data_prefix: Union[str, dict] = '',
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 **kwargs):
        self.gen_data_root = gen_data_root
        self.gen_data_file = gen_data_file
        self.split = split
        super().__init__(
            data_root=data_root,
            split=split,
            data_prefix=data_prefix,
            ann_file=ann_file,
            metainfo=metainfo,
            **kwargs)

    def load_data_list(self):
        """Load image paths and gt_labels."""
        if not self.ann_file:
            samples = self._find_samples()
        elif self.with_label:
            lines = list_from_file(self.ann_file)
            samples = [x.strip().rsplit(' ', 1) for x in lines]
        else:
            samples = list_from_file(self.ann_file)

        samples_gen = list_from_file(self.gen_data_file)
        samples_gen = set(
            [os.path.abspath(os.path.join(self.gen_data_root, sample)) for
             sample in samples_gen])

        self.num_gen_imgs = len(samples_gen)
        logger = MMLogger.get_current_instance()

        # Pre-build file backend to prevent verbose file backend inference.
        backend = get_file_backend(self.img_prefix, enable_singleton=True)
        data_list = []
        added_gen_imgs = 0
        for sample in samples:
            if self.with_label:
                filename, gt_label = sample
                img_path = backend.join_path(self.img_prefix, filename)
                info = {'img_path': img_path, 'gt_label': int(gt_label)}
            else:
                img_path = backend.join_path(self.img_prefix, sample)
                info = {'img_path': img_path}

            gen_img_path = img_path.replace(self.data_root, self.gen_data_root)
            if os.path.abspath(gen_img_path) in samples_gen:
                info['gen_img_path'] = gen_img_path
                added_gen_imgs += 1
            data_list.append(info)
        logger.info(f'Added {added_gen_imgs} generated images. '
                    f'Example gen_img_path is {gen_img_path}')
        assert added_gen_imgs > 0, RuntimeError(
            f'Generated images are not added to the training dataset. '
            f'Example gen_img_path is {gen_img_path}. '
            f'Please check the config.')
        return data_list

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Root of dataset: \t{self.data_root}',
            f'Root of generated dataset: \t{self.gen_data_root}',
            f'List of generated dataset: \t{self.gen_data_file}',
            f'Number of generated images: \t{self.num_gen_imgs}',
        ]
        return body

    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)
        return self.pipeline(data_info)


@DATASETS.register_module()
class ImageNetDiffusionScales(ImageNet):
    """Dataset class for ImageNet Diffusion.

    This dataset is an extension of the ImageNet dataset with added support
    for a generated image dataset. It loads image paths and optional generated
    image paths.

    The dataset supports two kinds of directory format,

    ::

        imagenet
        ├── train
        │   ├──class_x
        |   |   ├── x1.jpg
        |   |   ├── x2.jpg
        |   |   └── ...
        │   ├── class_y
        |   |   ├── y1.jpg
        |   |   ├── y2.jpg
        |   |   └── ...
        |   └── ...
        ├── val
        │   ├──class_x
        |   |   └── ...
        │   ├── class_y
        |   |   └── ...
        |   └── ...
        └── test
            ├── test1.jpg
            ├── test2.jpg
            └── ...

    or ::

        imagenet
        ├── train
        │   ├── x1.jpg
        │   ├── y1.jpg
        │   └── ...
        ├── val
        │   ├── x3.jpg
        │   ├── y3.jpg
        │   └── ...
        ├── test
        │   ├── test1.jpg
        │   ├── test2.jpg
        │   └── ...
        └── meta
            ├── train.txt
            └── val.txt


   Args:
        data_root (str): The root directory for the main dataset.
        gen_data_root (str): The root directory for the generated dataset.
        gen_data_file (str): Path to a file containing generated image paths.
        split (str): The dataset split, supports "train", "val", and "test".
        data_prefix (str | dict): Prefix for training data.
        ann_file (str): Annotation file path.
        metainfo (dict, optional): Meta information for the dataset.
        **kwargs: Other keyword arguments for CustomDataset and BaseDataset.

    Examples:
        >>> from mmpretrain.datasets import ImageNetDiffusion
        >>> train_dataset = ImageNetDiffusion(data_root='data/imagenet', 
                                              gen_data_root='data/gen_imagenet',
                                              gen_data_file='list.txt',
                                              split='train')
        >>> train_dataset
        Dataset ImageNetDiffusion
            Number of samples:  1281167
            Number of categories:  1000
            Number of generated images:  50000
            Root of main dataset:  data/imagenet
            Root of generated dataset:  data/gen_imagenet
    """  # noqa: E501

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    METAINFO = {'classes': IMAGENET_CATEGORIES}

    def __init__(self,
                 data_root: str = '',
                 gen_data_root: str = '',
                 gen_data_file: str = '',
                 split: str = '',
                 data_prefix: Union[str, dict] = '',
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 **kwargs):
        self.gen_data_root = gen_data_root
        self.gen_data_file = gen_data_file
        super().__init__(
            data_root=data_root,
            split=split,
            data_prefix=data_prefix,
            ann_file=ann_file,
            metainfo=metainfo,
            **kwargs)

    def load_data_list(self):
        """Load image paths and gt_labels."""
        if not self.ann_file:
            samples = self._find_samples()
        elif self.with_label:
            lines = list_from_file(self.ann_file)
            samples = [x.strip().rsplit(' ', 1) for x in lines]
        else:
            samples = list_from_file(self.ann_file)
        samples_gen_dict = dict_from_file(self.gen_data_file)
        samples_gen_set = set(
            [os.path.abspath(os.path.join(self.gen_data_root, sample)) for
             sample in samples_gen_dict.keys()])
        self.num_gen_imgs = len(samples_gen_set)
        logger = MMLogger.get_current_instance()

        # Pre-build file backend to prevent verbose file backend inference.
        backend = get_file_backend(self.img_prefix, enable_singleton=True)
        data_list = []
        added_gen_imgs = 0
        for sample in samples:
            if self.with_label:
                filename, gt_label = sample
                img_path = backend.join_path(self.img_prefix, filename)
                info = {'img_path': img_path, 'gt_label': int(gt_label)}
            else:
                img_path = backend.join_path(self.img_prefix, sample)
                info = {'img_path': img_path}
            gen_img_path = os.path.abspath(
                img_path.replace(self.data_root, self.gen_data_root))
            if gen_img_path in samples_gen_set and samples_gen_dict[sample] != "null":
                if isinstance(samples_gen_dict[sample], list):
                    info['gen_img_path'] = [os.path.join(self.gen_data_root, sample_scale) for sample_scale in samples_gen_dict[sample]]
                else:
                    info['gen_img_path'] = [os.path.join(self.gen_data_root, samples_gen_dict[sample])]

                added_gen_imgs += 1
            data_list.append(info)
        logger.info(f'Added {added_gen_imgs} generated images. '
                    f'Example gen_img_path is {gen_img_path}')
        assert added_gen_imgs > 0, RuntimeError(
            f'Generated images are not added to the training dataset. '
            f'Example gen_img_path is {gen_img_path}, {samples_gen_set.pop()}'
            f'Please check the config.')
        return data_list

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Root of dataset: \t{self.data_root}',
            f'Root of generated dataset: \t{self.gen_data_root}',
            f'List of generated dataset: \t{self.gen_data_file}',
            f'Number of generated images: \t{self.num_gen_imgs}',
        ]
        return body

    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)
        return self.pipeline(data_info)


@DATASETS.register_module()
class ImageNet21k(CustomDataset):
    """ImageNet21k Dataset.

    Since the dataset ImageNet21k is extremely big, contains 21k+ classes
    and 1.4B files. We won't provide the default categories list. Please
    specify it from the ``classes`` argument.
    The dataset directory structure is as follows,

    ImageNet21k dataset directory ::

        imagenet21k
        ├── train
        │   ├──class_x
        |   |   ├── x1.jpg
        |   |   ├── x2.jpg
        |   |   └── ...
        │   ├── class_y
        |   |   ├── y1.jpg
        |   |   ├── y2.jpg
        |   |   └── ...
        |   └── ...
        └── meta
            └── train.txt


    Args:
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (str | dict): Prefix for training data. Defaults to ''.
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        multi_label (bool): Not implement by now. Use multi label or not.
            Defaults to False.
        **kwargs: Other keyword arguments in :class:`CustomDataset` and
            :class:`BaseDataset`.

    Examples:
        >>> from mmpretrain.datasets import ImageNet21k
        >>> train_dataset = ImageNet21k(data_root='data/imagenet21k', split='train')
        >>> train_dataset
        Dataset ImageNet21k
            Number of samples:  14197088
            Annotation file:    data/imagenet21k/meta/train.txt
            Prefix of images:   data/imagenet21k/train
    """  # noqa: E501

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')

    def __init__(self,
                 data_root: str = '',
                 split: str = '',
                 data_prefix: Union[str, dict] = '',
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 multi_label: bool = False,
                 **kwargs):
        if multi_label:
            raise NotImplementedError(
                'The `multi_label` option is not supported by now.')
        self.multi_label = multi_label

        if split:
            splits = ['train']
            assert split in splits, \
                f"The split must be one of {splits}, but get '{split}'.\
                If you want to specify your own validation set or test set,\
                please set split to None."

            self.split = split
            data_prefix = split if data_prefix == '' else data_prefix

            if not ann_file:
                _ann_path = fileio.join_path(data_root, 'meta', f'{split}.txt')
                if fileio.exists(_ann_path):
                    ann_file = fileio.join_path('meta', f'{split}.txt')

        logger = MMLogger.get_current_instance()

        if not ann_file:
            logger.warning(
                'The ImageNet21k dataset is large, and scanning directory may '
                'consume long time. Considering to specify the `ann_file` to '
                'accelerate the initialization.')

        kwargs = {'extensions': self.IMG_EXTENSIONS, **kwargs}
        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            ann_file=ann_file,
            metainfo=metainfo,
            **kwargs)

        if self.CLASSES is None:
            logger.warning(
                'The CLASSES is not stored in the `ImageNet21k` class. '
                'Considering to specify the `classes` argument if you need '
                'do inference on the ImageNet-21k dataset')
