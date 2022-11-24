_base_ = [
    '../local_configs/_base_/models/segformer.py',
]

model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b4.pth',
    backbone=dict(
        type='mit_b4',
        style='pytorch'),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=21,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        decoder_params=dict(embed_dim=768)),
)

