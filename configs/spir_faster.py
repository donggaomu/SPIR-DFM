_base_ = [
     './_base_/schedules/schedule_1x.py','./_base_/default_runtime.py']
model = dict(
    type='MyFasterRCNN',
    backbone=dict(
        type='MyResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='/root/autodl-tmp/domain/configs/resnet50-0676ba61.pth')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=4,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100),
    ))

dataset_type = 'SUODACDataset'
classes = ('echinus','starfish','holothurian','scallop')
data_root = '../data/S-UODAC2020/'
img_norm_cfg = dict(
     mean=[0., 0., 0.], std=[255., 255., 255.], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='GeneratePuzzle',img_norm_cfg=img_norm_cfg,jig_classes = 30),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=False),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'img_puzzle', 'jig_labels']),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(512, 512),
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu = 4,
    workers_per_gpu = 4,
    train = (
            dict(
                type=dataset_type,
                classes=classes,
                ann_file=data_root + 'VOC2007/ImageSets/type1.txt',
                img_prefix=data_root + "VOC2007/",
                pipeline=train_pipeline),
            dict(
                type=dataset_type,
                classes=classes,
                ann_file=data_root + 'VOC2007/ImageSets/type2.txt',
                img_prefix=data_root + "VOC2007/",
                pipeline=train_pipeline),
            dict(
                type=dataset_type,
                classes=classes,
                ann_file=data_root + 'VOC2007/ImageSets/type3.txt',
                img_prefix=data_root + "VOC2007/",
                pipeline=train_pipeline),
            dict(
                type=dataset_type,
                classes=classes,
                ann_file=data_root + 'VOC2007/ImageSets/type4.txt',
                img_prefix=data_root + "VOC2007/",
                pipeline=train_pipeline),
            dict(
                type=dataset_type,
                classes=classes,
                ann_file=data_root + 'VOC2007/ImageSets/type5.txt',
                img_prefix=data_root + "VOC2007/",
                pipeline=train_pipeline),
            dict(
                type=dataset_type,
                classes=classes,
                ann_file=data_root + 'VOC2007/ImageSets/type6.txt',
                img_prefix=data_root + "VOC2007/",
                pipeline=train_pipeline),
    ),
    val=dict(
            type=dataset_type,
            classes=classes,
            ann_file=data_root + 'VOC2007/ImageSets/type7.txt',
            img_prefix=data_root + "VOC2007/",
            pipeline=test_pipeline),
    test=dict(
            type=dataset_type,
            classes=classes,
            ann_file=data_root + 'VOC2007/ImageSets/type7.txt',
            img_prefix=data_root + "VOC2007/",
            pipeline=test_pipeline)
    )

evaluation = dict(interval=1, metric='mAP')