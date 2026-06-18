load_from = None
resume = False
indices = None
backend_args = None
persistent_workers = True
crop_size = (64, 64)
psize=64
batch_size = 1
num_classes = 15
max_epochs =24
base_lr = 0.0001
num_workers = 8
log_level = 'INFO'
default_scope = 'mmdet'
dataset_type = 'UWDataset'
# dataset_type = 'USIS10KDataset'
# dataset_type = 'WHUInsSegDataset'
data_root = 'dataset/uw'
# data_root = 'dataset/usis'
# data_root = 'dataset/whu'
hf_sam_pretrain_name = "project/pretrain/sam-vit-base"
hf_sam_pretrain_ckpt_path = "project/pretrain/sam-vit-base/pytorch_model.bin"
work_dir = './work_dirs/sam_mask-rcnn_uw/our'
# work_dir = './work_dirs/hysam-uw/usis'
# work_dir = './work_dirs/hysam-uw/whu'
CLASSES = dict(classes=['Fish', 'Sea urchins', 'Sea cucumber', 'Sea turtle', 'Sea snake', 'Squid',
                    'Octopus', 'Shrimp', 'Ray', 'Shellfish', 'Seahorse', 'Starfish', 'Jellyfish',
                    'Diver', 'Coral'])
# CLASSES = dict(classes=['wrecks/ruins', 'fish', 'reefs', 'aquatic plants', 'human divers', 'robots', 'sea-floor'])
# CLASSES = dict(classes=['building'])

custom_imports = dict(imports=['project.model'], allow_failed_imports=False)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=6,
                    save_best=['coco/bbox_mAP', 'coco/segm_mAP'], rule='greater', save_last=True),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)


batch_augments = [dict(
    type='BatchFixedSizePad',
    size=crop_size,
    img_pad_value=0,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=False
)]

data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
    std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    bgr_to_rgb=True,
    pad_mask=True,
    pad_size_divisor=32,
    batch_augments=batch_augments
)


## ---------------------- MODEL ----------------------


model = dict(
    type='UWSAMMaskRCNN',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='UWSamVisionEncoder',
        image_size=psize,
        hf_pretrain_name=hf_sam_pretrain_name,
        extra_config=dict(output_hidden_states=True),
        init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path),
        peft_config=dict(
            peft_type="LORA",
            r=4,
            target_modules=["qkv"],
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
        )
    ),

    neck=dict(


        feature_spliter=dict(
            type='UWSimpleFPN',
            backbone_channel=256,
            in_channels=[64, 128, 256, 256],
            out_channels=256,
            num_outs=5,
            norm_cfg=dict(type='LN2d', requires_grad=True)),
    ),

    # neck=dict(
    #     in_channels=[256],
    #     num_outs=5,
    #     out_channels=256,
    #     type='FPN'),

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
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0)),
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
            num_classes=num_classes,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=num_classes,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    # model training and testing settings
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
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
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
            max_per_img=100,
            mask_thr_binary=0.5))
)

## ---------------------- dataset ----------------------


train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args, to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    # large scale jittering
    dict(
        type='RandomResize',
        scale=crop_size,
        ratio_range=(0.1, 2.0),
        resize_type='Resize',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=crop_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args, to_float32=True),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    dict(type='Pad', size=crop_size, pad_val=dict(img=(0.406 * 255, 0.456 * 255, 0.485 * 255), masks=0)),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor')
    )
]


train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        indices=indices,
        data_root=data_root,
        metainfo=CLASSES,
        ann_file='annotations/train.json',
        data_prefix=dict(img='train'),
        # filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args)
)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        indices=indices,
        data_root=data_root,
        metainfo=CLASSES,
        ann_file='annotations/val.json',
        data_prefix=dict(img='val'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args)
)

test_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        indices=indices,
        data_root=data_root,
        metainfo=CLASSES,
        ann_file='annotations/test.json',
        data_prefix=dict(img='test'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args)
)

val_evaluator = dict(
    type='CocoMetric',
    metric=['bbox', 'segm'],
    ann_file=data_root + '/annotations/val.json',
    format_only=False,
    backend_args=backend_args,
)

test_evaluator = dict(
    type='CocoMetric',
    metric=['bbox', 'segm'],
    ann_file=data_root + '/annotations/test.json',
    format_only=False,
    backend_args=backend_args,
)


## ---------------------- Optim ----------------------


train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

find_unused_parameters = True

# learning rate
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=50),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[10, 18],
        gamma=0.1
    )
]

optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='float16',
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        weight_decay=0.05,
    )
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=batch_size)


#default_hooks = dict(visualization=dict(type='VisualizationHook', interval=pretrain))# Customize interval as needed