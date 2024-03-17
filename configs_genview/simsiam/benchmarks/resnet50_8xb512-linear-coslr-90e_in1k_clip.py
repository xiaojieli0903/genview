_base_ = [
    '../../_base_/models/resnet50.py',
    '../../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../../_base_/schedules/imagenet_lars_coslr_90e.py',
    '../../_base_/default_runtime.py',
]

model = dict(
    backbone=dict(
        frozen_stages=4,
        init_cfg=dict(type='Pretrained', checkpoint='', prefix='backbone.')))

# dataset summary
train_dataloader = dict(batch_size=512, num_workers=10)

# runtime settings
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

data_preprocessor = dict(
    num_classes=1000,
    mean=[255 * 0.48145466, 255 * 0.4578275, 255 * 0.40821073],
    std=[255 * 0.26862954, 255 * 0.2613025, 255 * 0.27577711],
    to_rgb=True)
