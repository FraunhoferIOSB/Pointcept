_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 4  # bs: total bs in all gpus
num_worker = 32
mix_prob = 0.8
empty_cache = False
enable_amp = True
find_unused_parameters = True
clip_grad = 3.0
ignore_index=0
# trainer
train = dict(
    type="MultiDatasetTrainer",
)
# model settings
model = dict(
    type="PPT-v1m1",
    backbone=dict(
        type="PT-v3m1",
        in_channels=4,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        # enc_patch_size=(128, 128, 128, 128, 128),
        # enc_patch_size=(64, 64, 64, 64, 64),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        # dec_patch_size=(128, 128, 128, 128),
        # dec_patch_size=(64, 64, 64, 64),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=True,
        pdnorm_ln=True,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("car", "alice", "spot"),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=ignore_index),
    ],
    backbone_out_channels=64,
    context_channels=256,
    conditions=("car", "alice", "spot"),
    # num_classes=(8, 8, 8)
    template="[x]",
    clip_model="ViT-B/16",
    # fmt: off
    class_name=(
        "other",
        "artificial_structures",
        "artificial_ground",
        "natural_ground",
        "obstacle",
        "vehicle",
        "vegetation",
        "human",
        "sky"
    ),
    #  other (0), artificial_structures (1), artificial_ground (2), natural_ground (3), obstacle (4), vehicle (5), vegetation (6), human (7), sky (8)
    valid_index=(
        (0, 1, 2, 3, 4, 5, 6, 7),
        (0, 1, 2, 3, 4, 5, 6, 7),
        (0, 1, 2, 3, 4, 5, 6, 7),
    ),
    # fmt: on
    backbone_mode=False,
)

# scheduler settings
epoch = 50
eval_epoch = 50
optimizer = dict(type="AdamW", lr=0.0008, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.0008, 0.00005],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
param_dicts = [dict(keyword="block", lr=0.00005)]

# dataset settings
dataset_type = "GOOSEDataset"
# data_root = "/home/eb-xiaoya/datasets/goose"
# data_root = "/home/eb-xiaoya/datasets/livox_hap"
# data_root = "/home/eb-xiaoya/datasets/hesai"
data = dict(
    num_classes=8,
    ignore_index=ignore_index,
    # ignore_index = -1
    names = [
        "other",
        "artificial_structures",
        "artificial_ground",
        "natural_ground",
        "obstacle",
        "vehicle",
        "vegetation",
        "human",
    ],

        # data = dict(
        #     num_classes=8,
        #     ignore_index=ignore_index,
        #     names=names,
    train=dict(
        type="ConcatDataset",
        datasets=[
        dict(
            type=dataset_type,
            split=["train"],
            # split="train",
            data_root="/home/eb-xiaoya/datasets/goose_sep/car",
            transform=[
            # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1/6, 1/6], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1/6, 1/6], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment"),
                return_grid_coord=True,
            ),
            # dict(type="SphereCrop", point_max=1000000, mode="random"),
            # dict(type="CenterShift", apply_z=False),
            dict(type="Add", keys_dict={"condition": "car"}),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "condition"),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
        loop=1,
        ignore_index=ignore_index,
        ),
        dict(
                type=dataset_type,
                split=["train"], #, "val"
                # split="train",
                data_root="/home/eb-xiaoya/datasets/goose_sep/alice",
                transform=[
                # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
                # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
                dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
                dict(type="RandomRotate", angle=[-1/6, 1/6], axis="x", p=0.5),
                dict(type="RandomRotate", angle=[-1/6, 1/6], axis="y", p=0.5),
                dict(type="RandomScale", scale=[0.9, 1.1]),
                # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                dict(type="RandomFlip", p=0.5),
                dict(type="RandomJitter", sigma=0.005, clip=0.02),
                # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                dict(
                    type="GridSample",
                    grid_size=0.05,
                    hash_type="fnv",
                    mode="train",
                    keys=("coord", "strength", "segment"),
                    return_grid_coord=True,
                ),
                # dict(type="SphereCrop", point_max=1000000, mode="random"),
                # dict(type="CenterShift", apply_z=False),
                dict(type="Add", keys_dict={"condition": "alice"}),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "segment", "condition"),
                    feat_keys=("coord", "strength"),
                ),
            ],
            test_mode=False,
            ignore_index=ignore_index,
            loop=1,
        ),
        dict(
                type=dataset_type,
                split=["train"],
                # split="train",
                data_root="/home/eb-xiaoya/datasets/goose_sep/spot",
                transform=[
                # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
                # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
                dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
                dict(type="RandomRotate", angle=[-1/6, 1/6], axis="x", p=0.5),
                dict(type="RandomRotate", angle=[-1/6, 1/6], axis="y", p=0.5),
                dict(type="RandomScale", scale=[0.9, 1.1]),
                # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                dict(type="RandomFlip", p=0.5),
                dict(type="RandomJitter", sigma=0.005, clip=0.02),
                # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                dict(
                    type="GridSample",
                    grid_size=0.05,
                    hash_type="fnv",
                    mode="train",
                    keys=("coord", "strength", "segment"),
                    return_grid_coord=True,
                ),
                # dict(type="SphereCrop", point_max=1000000, mode="random"),
                # dict(type="CenterShift", apply_z=False),
                dict(type="Add", keys_dict={"condition": "spot"}),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "segment", "condition"),
                    feat_keys=("coord", "strength"),
                ),
            ],
            test_mode=False,
            ignore_index=ignore_index,
            loop=1,
        ),
            ]),
    val=
        dict(
        type=dataset_type,
        split="val",
        data_root="/home/eb-xiaoya/datasets/goose_sep/spot",
        transform=[
            # dict(type="PointClip", point_cloud_range=(-51.2, -51.2, -4, 51.2, 51.2, 2.4)),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment"),
                return_grid_coord=True,
            ),
            # dict(type="SphereCrop", point_max=1000000, mode='center'),
            dict(type="ToTensor"),
            dict(type="Add", keys_dict={"condition": "spot"}),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "condition"),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
        ),
    test=
        dict(
        type=dataset_type,
        split="val",
        data_root="/home/eb-xiaoya/datasets/goose_sep/spot",
        transform=[
                dict(type="Copy", keys_dict={"segment": "origin_segment"}),
                dict(
                    type="GridSample",
                    grid_size=0.025,
                    hash_type="fnv",
                    mode="train",
                    keys=("coord", "strength", "segment"),
                    return_inverse=True,
                ),
            ],
        test_mode=True,
        ignore_index=ignore_index,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                keys=("coord", "strength"),
            ),
            crop=None,
            post_transform=[
                dict(type="Add", keys_dict={"condition": "spot"}),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index", "condition"),
                    feat_keys=("coord", "strength"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[1, 1])],
                
                ],
            ),
        ),


    # val=dict(
    #     type="ConcatDataset",
    #     datasets=[
    #     dict(
    #         type=dataset_type,
    #         split=["val"],
    #         # split="val",
    #         data_root="/home/eb-xiaoya/datasets/goose_sep/car",
    #         transform=[
    #             # dict(type="PointClip", point_cloud_range=(-51.2, -51.2, -4, 51.2, 51.2, 2.4)),
    #             dict(
    #                 type="GridSample",
    #                 grid_size=0.05,
    #                 hash_type="fnv",
    #                 mode="train",
    #                 keys=("coord", "strength", "segment"),
    #                 return_grid_coord=True,
    #             ),
    #             # dict(type="SphereCrop", point_max=1000000, mode='center'),
    #             dict(type="ToTensor"),
    #             dict(type="Add", keys_dict={"condition": "car"}),
    #             dict(
    #                 type="Collect",
    #                 keys=("coord", "grid_coord", "segment", "condition"),
    #                 feat_keys=("coord", "strength"),
    #             ),
    #         ],
    #         test_mode=False,
    #         ignore_index=ignore_index,
    #     ),
    #     dict(
    #         type=dataset_type,
    #         split=["val"],
    #         # split="val",
    #         data_root="/home/eb-xiaoya/datasets/goose_sep/alice",
    #         transform=[
    #             # dict(type="PointClip", point_cloud_range=(-51.2, -51.2, -4, 51.2, 51.2, 2.4)),
    #             dict(
    #                 type="GridSample",
    #                 grid_size=0.05,
    #                 hash_type="fnv",
    #                 mode="train",
    #                 keys=("coord", "strength", "segment"),
    #                 return_grid_coord=True,
    #             ),
    #             # dict(type="SphereCrop", point_max=1000000, mode='center'),
    #             dict(type="ToTensor"),
    #             dict(type="Add", keys_dict={"condition": "alice"}),
    #             dict(
    #                 type="Collect",
    #                 keys=("coord", "grid_coord", "segment", "condition"),
    #                 feat_keys=("coord", "strength"),
    #             ),
    #         ],
    #         test_mode=False,
    #         ignore_index=ignore_index,
    #     ),
    #     dict(
    #         type=dataset_type,
    #         split=["val"],
    #         # split="val",
    #         data_root="/home/eb-xiaoya/datasets/goose_sep/spot",
    #         transform=[
    #             # dict(type="PointClip", point_cloud_range=(-51.2, -51.2, -4, 51.2, 51.2, 2.4)),
    #             dict(
    #                 type="GridSample",
    #                 grid_size=0.05,
    #                 hash_type="fnv",
    #                 mode="train",
    #                 keys=("coord", "strength", "segment"),
    #                 return_grid_coord=True,
    #             ),
    #             # dict(type="SphereCrop", point_max=1000000, mode='center'),
    #             dict(type="ToTensor"),
    #             dict(type="Add", keys_dict={"condition": "spot"}),
    #             dict(
    #                 type="Collect",
    #                 keys=("coord", "grid_coord", "segment", "condition"),
    #                 feat_keys=("coord", "strength"),
    #             ),
    #         ],
    #         test_mode=False,
    #         ignore_index=ignore_index,
    #     ),
    #     ],
    #     )
)
    # ,
    # test=
    #     dict(
    #     type="ConcatDataset",
    #     datasets=[
    #     dict(
    #         type=dataset_type,
    #         split=["test"],
    #         # split="test",
    #         data_root="/home/eb-xiaoya/datasets/goose_sep/car",
    #         transform=[
    #             dict(type="Copy", keys_dict={"segment": "origin_segment"}),
    #             dict(
    #                 type="GridSample",
    #                 grid_size=0.025,
    #                 hash_type="fnv",
    #                 mode="train",
    #                 keys=("coord", "strength", "segment"),
    #                 return_inverse=True,
    #             ),
    #         ],
    #         test_mode=True,
    #         test_cfg=dict(
    #             voxelize=dict(
    #                 type="GridSample",
    #                 grid_size=0.05,
    #                 hash_type="fnv",
    #                 mode="test",
    #                 return_grid_coord=True,
    #                 keys=("coord", "strength"),
    #             ),
    #             crop=None,
    #             post_transform=[
    #                 dict(type="Add", keys_dict={"condition": "car"}),
    #                 dict(type="ToTensor"),
    #                 dict(
    #                     type="Collect",
    #                     keys=("coord", "grid_coord", "index", "condition"),
    #                     feat_keys=("coord", "strength"),
    #                 ),
    #             ],
    #             aug_transform=[
    #                 [dict(type="RandomScale", scale=[0.9, 0.9])],
    #                 [dict(type="RandomScale", scale=[0.95, 0.95])],
    #                 [dict(type="RandomScale", scale=[1, 1])],
    #                 [dict(type="RandomScale", scale=[1.05, 1.05])],
    #                 [dict(type="RandomScale", scale=[1.1, 1.1])],
    #                 [
    #                     dict(type="RandomScale", scale=[0.9, 0.9]),
    #                     dict(type="RandomFlip", p=1),
    #                 ],
    #                 [
    #                     dict(type="RandomScale", scale=[0.95, 0.95]),
    #                     dict(type="RandomFlip", p=1),
    #                 ],
    #                 [dict(type="RandomScale", scale=[1, 1]), dict(type="RandomFlip", p=1)],
    #                 [
    #                     dict(type="RandomScale", scale=[1.05, 1.05]),
    #                     dict(type="RandomFlip", p=1),
    #                 ],
    #                 [
    #                     dict(type="RandomScale", scale=[1.1, 1.1]),
    #                     dict(type="RandomFlip", p=1),
    #                 ],
    #             ],
    #         ),
    #     ),
    #     dict(
    #         type=dataset_type,
    #         split=["test"],
    #         # split="test",
    #         data_root="/home/eb-xiaoya/datasets/goose_sep/spot",
    #         transform=[
    #             dict(type="Copy", keys_dict={"segment": "origin_segment"}),
    #             dict(
    #                 type="GridSample",
    #                 grid_size=0.025,
    #                 hash_type="fnv",
    #                 mode="train",
    #                 keys=("coord", "strength", "segment"),
    #                 return_inverse=True,
    #             ),
    #         ],
    #         test_mode=True,
    #         test_cfg=dict(
    #             voxelize=dict(
    #                 type="GridSample",
    #                 grid_size=0.05,
    #                 hash_type="fnv",
    #                 mode="test",
    #                 return_grid_coord=True,
    #                 keys=("coord", "strength"),
    #             ),
    #             crop=None,
    #             post_transform=[
    #                 dict(type="Add", keys_dict={"condition": "spot"}),
    #                 dict(type="ToTensor"),
    #                 dict(
    #                     type="Collect",
    #                     keys=("coord", "grid_coord", "index", "condition"),
    #                     feat_keys=("coord", "strength"),
    #                 ),
    #             ],
    #             aug_transform=[
    #                 [dict(type="RandomScale", scale=[0.9, 0.9])],
    #                 [dict(type="RandomScale", scale=[0.95, 0.95])],
    #                 [dict(type="RandomScale", scale=[1, 1])],
    #                 [dict(type="RandomScale", scale=[1.05, 1.05])],
    #                 [dict(type="RandomScale", scale=[1.1, 1.1])],
    #                 [
    #                     dict(type="RandomScale", scale=[0.9, 0.9]),
    #                     dict(type="RandomFlip", p=1),
    #                 ],
    #                 [
    #                     dict(type="RandomScale", scale=[0.95, 0.95]),
    #                     dict(type="RandomFlip", p=1),
    #                 ],
    #                 [dict(type="RandomScale", scale=[1, 1]), dict(type="RandomFlip", p=1)],
    #                 [
    #                     dict(type="RandomScale", scale=[1.05, 1.05]),
    #                     dict(type="RandomFlip", p=1),
    #                 ],
    #                 [
    #                     dict(type="RandomScale", scale=[1.1, 1.1]),
    #                     dict(type="RandomFlip", p=1),
    #                 ],
    #             ],
    #         ),
    #     ),
        
    #     dict(
    #         type=dataset_type,
    #         split=["test"],
    #         # split="test",
    #         data_root="/home/eb-xiaoya/datasets/goose_sep/alice",
    #         transform=[
    #             dict(type="Copy", keys_dict={"segment": "origin_segment"}),
    #             dict(
    #                 type="GridSample",
    #                 grid_size=0.025,
    #                 hash_type="fnv",
    #                 mode="train",
    #                 keys=("coord", "strength", "segment"),
    #                 return_inverse=True,
    #             ),
    #         ],
    #         test_mode=True,
    #         test_cfg=dict(
    #             voxelize=dict(
    #                 type="GridSample",
    #                 grid_size=0.05,
    #                 hash_type="fnv",
    #                 mode="test",
    #                 return_grid_coord=True,
    #                 keys=("coord", "strength"),
    #             ),
    #             crop=None,
    #             post_transform=[
    #                 dict(type="Add", keys_dict={"condition": "alice"}),
    #                 dict(type="ToTensor"),
    #                 dict(
    #                     type="Collect",
    #                     keys=("coord", "grid_coord", "index", "condition"),
    #                     feat_keys=("coord", "strength"),
    #                 ),
    #             ],
    #             aug_transform=[
    #                 [dict(type="RandomScale", scale=[0.9, 0.9])],
    #                 [dict(type="RandomScale", scale=[0.95, 0.95])],
    #                 [dict(type="RandomScale", scale=[1, 1])],
    #                 [dict(type="RandomScale", scale=[1.05, 1.05])],
    #                 [dict(type="RandomScale", scale=[1.1, 1.1])],
    #                 [
    #                     dict(type="RandomScale", scale=[0.9, 0.9]),
    #                     dict(type="RandomFlip", p=1),
    #                 ],
    #                 [
    #                     dict(type="RandomScale", scale=[0.95, 0.95]),
    #                     dict(type="RandomFlip", p=1),
    #                 ],
    #                 [dict(type="RandomScale", scale=[1, 1]), dict(type="RandomFlip", p=1)],
    #                 [
    #                     dict(type="RandomScale", scale=[1.05, 1.05]),
    #                     dict(type="RandomFlip", p=1),
    #                 ],
    #                 [
    #                     dict(type="RandomScale", scale=[1.1, 1.1]),
    #                     dict(type="RandomFlip", p=1),
    #                 ],
    #             ],
    #         ),
    #     ),
    #     ]
    #     )
    #     )
