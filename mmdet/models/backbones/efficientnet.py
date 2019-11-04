import logging

import torch.nn as nn
from ..registry import BACKBONES
from ..utils import build_conv_layer


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Block
    """

    def __init__(self,
                 inplanes,
                 outplanes,
                 expansion,
                 stride=1,
                 style='pytorch',
                 conv_cfg=None,
                 se_cfg=None):
        super(MBConvBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = outplanes
        self.expansion = expansion
        self.stride = stride
        self.style = style
        self.use_res_connect = stride == 1 and inplanes == outplanes

        expplanes = self.expansion * inplanes

        # TODO: Add support for other styles
        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
            self.conv3_stride = 1

        # TODO: verify the bias term value
        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            expplanes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False
        )

        # TODO: verify the bias term value
        self.conv2 = build_conv_layer(
            conv_cfg,
            expplanes,
            expplanes,
            kernel_size=3,
            stride=self.conv2_stride,
            bias=False
        )

        # TODO: verify the bias term value
        self.conv3 = build_conv_layer(
            conv_cfg,
            expplanes,
            outplanes,
            kernel_size=1,
            stride=self.conv3_stride,
            bias=False
        )

        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu6(out)

        out = self.conv2(out)
        out = self.relu6(out)

        out = self.conv3(out)

        if self.use_res_connect:
            out += identity

        return out


@BACKBONES.register_module
class EfficientNet(nn.Module):
    """The class implements EfficientNet.
    """

    def __init__(self, param):
        super(EfficientNet, self).__init__()
        self.param = param

    def forward(self, x):
        return x
