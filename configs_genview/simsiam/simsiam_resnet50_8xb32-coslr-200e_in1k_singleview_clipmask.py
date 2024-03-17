_base_ = [
    '../_base_/datasets/imagenet_bs32_mocov2.py',
    '../_base_/schedules/imagenet_sgd_coslr_200e.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='SimSiam',
    use_clip_mask=True,
    clip_model_name='convnext_base_w',
    adjust_loss=True,
    backbone=dict(
        type='ResNet',
        depth=50,
        norm_cfg=dict(type='SyncBN'),
        zero_init_residual=True),
    neck=dict(
        type='NonLinearNeck',
        in_channels=2048,
        hid_channels=2048,
        out_channels=2048,
        num_layers=3,
        with_last_bn_affine=False,
        with_avg_pool=True),
    head=dict(
        type='LatentPredictHead',
        loss=dict(type='CosineSimilarityLoss'),
        predictor=dict(
            type='NonLinearNeck',
            in_channels=2048,
            hid_channels=512,
            out_channels=2048,
            with_avg_pool=False,
            with_last_bn=False,
            with_last_bias=True)),
)

# optimizer
# set base learning rate
lr = 0.05
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=lr, weight_decay=1e-4, momentum=0.9),
    paramwise_cfg=dict(custom_keys={'predictor': dict(fix_lr=True)}))

# runtime settings
default_hooks = dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

# additional hooks
custom_hooks = [
    dict(type='SimSiamHook', priority='HIGH', fix_pred_lr=True, lr=lr)
]

# dataset settings
# The difference between mocov2 and mocov1 is the transforms in the pipeline
view_pipeline = [
    dict(
        type='RandomResizedCrop',
        scale=224,
        crop_ratio_range=(0.2, 1.),
        backend='pillow'),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1)
        ],
        prob=0.8),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)),
    dict(
        type='GaussianBlur',
        magnitude_range=(0.1, 2.0),
        magnitude_std='inf',
        prob=0.5),
    dict(type='RandomFlip', prob=0.5),
]

train_pipeline = [
    dict(type='LoadImageFromFileDiffusion'),
    dict(type='MultiView', num_views=2, transforms=[view_pipeline], use_gen_img=True, prob_gen_img=1.0),
    dict(type='PackInputs', meta_keys=('sample_idx', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'gen_img_path', 'gen_img_scale', 'gen_chosen_view'))
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    drop_last=True,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='ImageNetDiffusion',
        data_root='data/imagenet/',
        split='train',
        gen_data_root='data/imagenet/train_variations/',
        gen_data_file='data/imagenet/train_variations.txt',
        pipeline=train_pipeline))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=5),
    dist_cfg=dict(backend='nccl'))

data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[255 * 0.48145466, 255 * 0.4578275, 255 * 0.40821073],
    std=[255 * 0.26862954, 255 * 0.2613025, 255 * 0.27577711],
    to_rgb=True)
