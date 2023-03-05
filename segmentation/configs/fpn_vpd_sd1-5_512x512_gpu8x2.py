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

data = dict(samples_per_gpu=2, workers_per_gpu=8)
