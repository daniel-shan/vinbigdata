dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.05,
        rotate_limit=10,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(2000, 720), (2000, 1440)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='RandomFlip',
        flip_ratio=0.0),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2000, 1440),        
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        ann_file='/home/ec2-user/input/train_annotations.json',
        img_prefix='/home/ec2-user/input/',
        pipeline=train_pipeline,
        classes=('Aortic_enlargement', 'Atelectasis', 'Calcification',
                 'Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration',
                 'Lung_Opacity', 'Nodule/Mass', 'Other_lesion',
                 'Pleural_effusion', 'Pleural_thickening', 'Pneumothorax',
                 'Pulmonary_fibrosis')),
    val=dict(
        type='CocoDataset',
        ann_file='/home/ec2-user/input/val_annotations.json',
        img_prefix='/home/ec2-user/input/',
        pipeline=test_pipeline,
        classes=('Aortic_enlargement', 'Atelectasis', 'Calcification',
                 'Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration',
                 'Lung_Opacity', 'Nodule/Mass', 'Other_lesion',
                 'Pleural_effusion', 'Pleural_thickening', 'Pneumothorax',
                 'Pulmonary_fibrosis')),
    test=dict(
        type='CocoDataset',
        ann_file='/kaggle/input/test_annotations.json',
        img_prefix='/kaggle/input/',
        pipeline=test_pipeline,
        classes=('Aortic_enlargement', 'Atelectasis', 'Calcification',
                 'Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration',
                 'Lung_Opacity', 'Nodule/Mass', 'Other_lesion',
                 'Pleural_effusion', 'Pleural_thickening', 'Pneumothorax',
                 'Pulmonary_fibrosis')))
evaluation = dict(interval=1, metric='bbox')
optimizer = dict(
    type='SGD',
    lr=0.115,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(bias_lr_mult=2.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2400,
    warmup_ratio=0.1,
    step=[16, 22])
total_epochs = 48
checkpoint_config = dict(interval=12)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/home/ec2-user/checkpoints/vfnet.pth'
resume_from = None
workflow = [('train', 1)]
model = dict(
    type='VFNet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='VFNetHead',
        num_classes=14,
        in_channels=256,
        stacked_convs=3,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=False,
        dcn_on_last_conv=True,
        use_atss=True,
        use_vfl=True,
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.5),
        loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0)))
train_cfg = dict(
    assigner=dict(type='ATSSAssigner', topk=9),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.01,
    nms=dict(type='nms', iou_threshold=0.5),
    max_per_img=200)
classes = ('Aortic_enlargement', 'Atelectasis', 'Calcification',
           'Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration',
           'Lung_Opacity', 'Nodule/Mass', 'Other_lesion', 'Pleural_effusion',
           'Pleural_thickening', 'Pneumothorax', 'Pulmonary_fibrosis')
seed = 42
gpu_ids = range(0, 1)
work_dir = '/home/ec2-user/vinbig_output'
