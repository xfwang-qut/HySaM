from typing import List

import einops
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import MultiConfig, OptConfigType


@MODELS.register_module()
class UWFPN(BaseModule):
    def __init__(
            self,
            feature_aggregator=None,
            feature_spliter=None,
            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        if feature_aggregator is not None:
            self.feature_aggregator = MODELS.build(feature_aggregator)
        if feature_spliter is not None:
            self.feature_spliter = MODELS.build(feature_spliter)

    def forward(self, inputs):
        if hasattr(self, 'feature_aggregator'):
            x = self.feature_aggregator(inputs)
        else:
            x = inputs
        if hasattr(self, 'feature_spliter'):
            x = self.feature_spliter(x)
        else:
            x = (x,)
        return x


@MODELS.register_module()
class UWFeatureFusion(BaseModule):
    in_channels_dict = {
        'base': [768] * (12 + 1),
        'large': [1024] * (24 + 1),
        'huge': [1280] * (32 + 1),
    }

    def __init__(self, in_channels, hidden_channels=256, out_channels=256, select_layers=range(1, 12, 2), init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        model_arch = 'base' if 'base' in in_channels else 'large' if 'large' in in_channels else 'huge'
        self.in_channels = self.in_channels_dict[model_arch]
        max_layer = len(self.in_channels) - 1
        self.select_layers = [i for i in select_layers if i <= max_layer]
        self.upbottom_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
                nn.GroupNorm(32, hidden_channels),
                nn.ReLU(inplace=True)
            ) for in_channels in self.in_channels
        ])

        self.weight_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
                nn.GroupNorm(32, hidden_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(len(self.in_channels) - 1)
        ])

        # self.lateral_conv = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=pretrain),
        #         nn.GroupNorm(32, hidden_channels),
        #         nn.ReLU(inplace=True)
        #     ) for _ in self.in_channels
        # ])

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            )

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels), f"输入特征图数量 {len(inputs)} 与模型期望的数量 {len(self.in_channels)} 不匹配。"

        if inputs[0].shape[1] != self.in_channels[0]:  # 检查通道数是否匹配
            inputs = [einops.rearrange(x, 'b h w c -> b c h w') for x in inputs]
        feature_maps = inputs[::-1]
        up_bottom_features = [self.upbottom_conv[0](feature_maps[0])]
        for i, feature in enumerate(feature_maps[1:]):
            down_feature = self.upbottom_conv[i + 1](feature)
            high_feature = up_bottom_features[i]
            _, _, h, w = feature.shape
            if high_feature.shape[-1] != w or high_feature.shape[-2] != h:
                high_feature = F.interpolate(high_feature, size=(h, w), mode='bilinear', align_corners=False)
            high_weighted = self.weight_conv[i](high_feature)
            fusion_feature = high_weighted * down_feature + high_feature
            up_bottom_features.append(fusion_feature)
        results = up_bottom_features[::-1]
        # for i in range(len(results)):
        #     results[i] = self.lateral_conv[i](results[i])
        output = self.fusion_conv(results[0])
        return output


@MODELS.register_module()
class UWSimpleFPN(BaseModule):
    def __init__(self,
                 backbone_channel: int,
                 in_channels: List[int],
                 out_channels: int,
                 num_outs: int,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 act_cfg: OptConfigType = None,
                 init_cfg: MultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.backbone_channel = backbone_channel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel,
                               self.backbone_channel // 2, 2, 2),
            build_norm_layer(norm_cfg, self.backbone_channel // 2)[1],
            nn.GELU(),
            nn.ConvTranspose2d(self.backbone_channel // 2,
                               self.backbone_channel // 4, 2, 2))
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel,
                               self.backbone_channel // 2, 2, 2))
        self.fpn3 = nn.Sequential(nn.Identity())
        self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.num_ins):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, input: Tensor) -> tuple:
        """Forward function.

        Args:
            inputs (Tensor): Features from the upstream network, 4D-tensor
        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        # build FPN
        inputs = []
        inputs.append(self.fpn1(input))
        inputs.append(self.fpn2(input))
        inputs.append(self.fpn3(input))
        inputs.append(self.fpn4(input))

        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        outs = [self.fpn_convs[i](laterals[i]) for i in range(self.num_ins)]

        if self.num_outs > len(outs):
            for i in range(self.num_outs - self.num_ins):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return tuple(outs)


# @MODELS.register_module()
# class UWFeatureFusion(BaseModule):
#     in_channels_dict = {
#         'base': [768] * (12 + 1),
#         'large': [1024] * (24 + 1),
#         'huge': [1280] * (32 + 1),
#     }
#
#     def __init__(
#             self,
#             in_channels,
#             hidden_channels=256,
#             out_channels=256,
#             init_cfg=None,
#             select_layers=range(1, 12, 2),
#     ):
#         super().__init__(init_cfg=init_cfg)
#         model_arch = 'base' if 'base' in in_channels else 'large' if 'large' in in_channels else 'huge'
#         self.in_channels = [self.in_channels_dict[model_arch][i] for i in select_layers]  # 仅选取指定层
#         self.select_layers = select_layers
#
#         self.DownConv = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
#                 nn.GroupNorm(32, hidden_channels),
#                 nn.ReLU(inplace=True)
#             ) for in_channels in self.in_channels
#         ])
#
#         self.WeightConv = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
#                 nn.GroupNorm(32, hidden_channels),
#                 nn.ReLU(inplace=True)
#             ) for _ in range(len(self.in_channels) - 1)
#         ])
#
#         self.FusionConv = nn.Sequential(
#             nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
#             nn.GroupNorm(32, out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.GroupNorm(32, out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#         )
#
#     def forward(self, inputs):
#         selected_inputs = [inputs[i] for i in self.select_layers]  # 仅保留选定层
#         assert len(selected_inputs) == len(self.in_channels), f"输入特征图数量 {len(selected_inputs)} 与模型期望的数量 {len(self.in_channels)} 不匹配。"
#
#         if selected_inputs[0].shape[1] != self.in_channels[0]:
#             selected_inputs = [einops.rearrange(x, 'b h w c -> b c h w') for x in selected_inputs]
#
#         feature_maps = selected_inputs[::-1]
#         features = [self.DownConv[0](feature_maps[0])]
#         for i, feature in enumerate(feature_maps[1:]):
#             down_feature = self.DownConv[i + 1](feature)
#             high_feature = features[i]
#             _, _, h, w = feature.shape
#             if high_feature.shape[-1] != w or high_feature.shape[-2] != h:
#                 high_feature = F.interpolate(high_feature, size=(h, w), mode='bilinear', align_corners=False)
#             high_weighted = self.WeightConv[i](high_feature)
#             fusion_feature = high_weighted * down_feature + high_feature
#             features.append(fusion_feature)
#         results = features[::-1]
#         output = self.FusionConv(results[0])
#         return output


# @MODELS.register_module()
# class UWFeatureFusion(BaseModule):
#     in_channels_dict = {
#         'base': [768] * (12 + 1),
#         'large': [1024] * (24 + 1),
#         'huge': [1280] * (32 + 1),
#     }
#
#     def __init__(
#             self,
#             in_channels,
#             hidden_channels=256,
#             out_channels=256,
#             init_cfg=None,
#     ):
#         super().__init__(init_cfg=init_cfg)
#         model_arch = 'base' if 'base' in in_channels else 'large' if 'large' in in_channels else 'huge'
#         self.in_channels = self.in_channels_dict[model_arch]
#         self.DownConv = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
#                 nn.GroupNorm(32, hidden_channels),
#                 nn.ReLU(inplace=True)
#             ) for in_channels in self.in_channels
#         ])
#
#         self.WeightConv = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
#                 nn.GroupNorm(32, hidden_channels),
#                 nn.ReLU(inplace=True)
#             ) for _ in range(len(self.in_channels) - 1)
#         ])
#
#         self.FusionConv = nn.Sequential(
#             nn.Conv2d(hidden_channels, out_channels, kernel_size=1, padding=1),
#             nn.GroupNorm(32, out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.GroupNorm(32, out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             )
#
#     def forward(self, inputs):
#         assert len(inputs) == len(self.in_channels), f"输入特征图数量 {len(inputs)} 与模型期望的数量 {len(self.in_channels)} 不匹配。"
#
#         if inputs[0].shape[1] != self.in_channels[0]:  # 检查通道数是否匹配
#             inputs = [einops.rearrange(x, 'b h w c -> b c h w') for x in inputs]
#         feature_maps = inputs[::-1]
#         features = [self.DownConv[0](feature_maps[0])]
#         for i, feature in enumerate(feature_maps[1:]):
#             down_feature = self.DownConv[i + 1](feature)
#             high_feature = features[i]
#             _, _, h, w = feature.shape
#             if high_feature.shape[-1] != w or high_feature.shape[-2] != h:
#                 high_feature = F.interpolate(high_feature, size=(h, w), mode='bilinear', align_corners=False)
#             high_weighted = self.WeightConv[i](high_feature)
#             fusion_feature = high_weighted * down_feature + high_feature
#             features.append(fusion_feature)
#         results = features[::-1]
#         output = self.FusionConv(results[0])
#         return output