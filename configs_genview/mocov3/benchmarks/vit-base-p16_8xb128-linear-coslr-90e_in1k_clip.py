_base_ = [
    '../../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../../_base_/default_runtime.py',
]

# dataset settings
train_dataloader = dict(batch_size=128)

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MoCoV3ViT',
        arch='base',  # embed_dim = 768
        img_size=224,
        patch_size=16,
        stop_grad_conv1=True,
        frozen_stages=12,
        init_cfg=dict(type='Pretrained', checkpoint='', prefix='backbone.'),
        norm_eval=True),
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=1000,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        init_cfg=dict(type='Normal', std=0.01, layer='Linear'),
    ))

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=12, momentum=0.9, weight_decay=0.))

# learning rate scheduler
param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=90, by_epoch=True, begin=0, end=90)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=90)
val_cfg = dict()
test_cfg = dict()

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=3))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

data_preprocessor = dict(
    num_classes=1000,
    mean=[
        122.7709383,
        116.7460125,
        104.09373615000001,
    ],
    std=[
        68.5005327,
        66.6321375,
        70.32316304999999,
    ],
    to_rgb=True)
