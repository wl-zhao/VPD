_base_ = [
    '_base_/models/fpn_r50.py', '_base_/datasets/ade20k_vpd.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_80k.py'
]

model = dict(
    type='VPDSeg',
    sd_path='checkpoints/v1-5-pruned-emaonly.ckpt',
    neck=dict(
        type='FPN',
        in_channels=[320, 790, 1430, 1280],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='FPNHead',
        num_classes=150,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
)

lr_config = dict(policy='poly', power=1, min_lr=0.0, by_epoch=False,
                warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6)


optimizer = dict(type='AdamW', lr=0.00008, weight_decay=0.001,
        paramwise_cfg=dict(custom_keys={'unet': dict(lr_mult=0.1),
                                        'encoder_vq': dict(lr_mult=0.0),
                                        'text_encoder': dict(lr_mult=0.0),
                                        'norm': dict(decay_mult=0.)}))

data = dict(samples_per_gpu=2, workers_per_gpu=8)
