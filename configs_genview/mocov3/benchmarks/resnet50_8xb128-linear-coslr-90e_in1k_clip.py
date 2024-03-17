_base_ = [
    '../../_base_/models/resnet50.py',
    '../../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../../_base_/schedules/imagenet_sgd_coslr_100e.py',
    '../../_base_/default_runtime.py',
]

# dataset settings
train_dataloader = dict(batch_size=128, num_workers=10, persistent_workers=True, pin_memory=True)

#model = dict(backbone=dict(frozen_stages=4, norm_eval=True))
model = dict(
    backbone=dict(
        frozen_stages=4,
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='', prefix='backbone.')))

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.4, momentum=0.9, weight_decay=0.),
)

# learning rate scheduler
param_scheduler = [
   dict(type='CosineAnnealingLR', T_max=90, by_epoch=True, begin=0, end=90)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=90)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2))

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
