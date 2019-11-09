import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from mmcv.runner import load_checkpoint
from mmcv.cnn import constant_init, kaiming_init
from collections import namedtuple
from ..registry import BACKBONES


field_names = ['kernel_size', 'num_repeat', 'inplanes', 'outplanes',
               'expansion', 'id_skip', 'stride', 'se_ratio']
StageArgs = namedtuple('StageArgs', field_names)
StageArgs.__new__.__defaults__ = (None,) * len(StageArgs._fields)


def drop_connect(inputs, p):
    """Drop connect.
    """
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype,
                                device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


class Conv2dStaticSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a fixed image size"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 image_size=None,
                 **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        if len(self.stride) == 2:
            self.stride = self.stride
        else:
            self.stride = [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size] * 2
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0]
                    + (kh - 1) * self.dilation[0]
                    + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1]
                    + (kw - 1) * self.dilation[1]
                    + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d(
                (pad_w // 2, pad_w - pad_w // 2,
                 pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding,
                     self.dilation, self.groups)
        return x


def get_same_padding_conv2d(image_size):
    """ Chooses static padding if you have specified an image size.
        Static padding is necessary for ONNX exporting of models.
    """

    return partial(Conv2dStaticSamePadding, image_size=image_size)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Block
    """

    def __init__(self,
                 image_size,
                 inplanes,
                 outplanes,
                 kernel_size,
                 expansion,
                 stride,
                 momentum,
                 eps,
                 se_ratio=None,
                 style='pytorch',
                 conv_cfg=None):
        super(MBConvBlock, self).__init__()
        assert (se_ratio is None) or (0 < se_ratio <= 1)
        self.has_se = (se_ratio is not None) and (0 < se_ratio <= 1)
        self.image_size = image_size
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.kernel_size = kernel_size
        self.expansion = expansion
        self.stride = stride
        self.momentum = 1 - momentum
        self.eps = eps
        self.se_ratio = se_ratio
        self.id_skip = stride == 1 and self.inplanes == self.outplanes
        self.style = style

        Conv2d = get_same_padding_conv2d(image_size=self.image_size)
        expplanes = self.expansion * self.inplanes

        # TODO: Add support for other styles
        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
            self.conv3_stride = 1

        if self.expansion != 1:
            self._expand_conv = Conv2d(
                self.inplanes,
                expplanes,
                1,
                stride=self.conv1_stride,
                bias=False
            )

            self._bn0 = nn.BatchNorm2d(expplanes, self.eps, self.momentum)

        self._depthwise_conv = Conv2d(
            expplanes,
            expplanes,
            self.kernel_size,
            stride=self.conv2_stride,
            bias=False,
            groups=expplanes
        )

        self._bn1 = nn.BatchNorm2d(expplanes, self.eps, self.momentum)

        if self.has_se:
            squeeze_planes = max(1, int(self.inplanes * self.se_ratio))
            self._se_reduce = Conv2d(expplanes, squeeze_planes, 1)
            self._se_expand = Conv2d(squeeze_planes, expplanes, 1)

        self._project_conv = Conv2d(
            expplanes,
            self.outplanes,
            1,
            stride=self.conv3_stride,
            bias=False
        )

        self._bn2 = nn.BatchNorm2d(self.outplanes, self.eps, self.momentum)
        self._swish = Swish()
        self.identity = nn.Identity()

    def forward(self, x, drop_connect_rate):
        identity = self.identity(x)
        out = x

        # Expansion phase
        if self.expansion != 1:
            out = self._expand_conv(out)
            out = self._bn0(out)

        # Depthwise phase
        out = self._depthwise_conv(out)
        out = self._bn1(out)
        out = self._swish(out)

        # Squeeze and excitation phase
        if self.has_se:
            out_squeezed = F.adaptive_avg_pool2d(out, 1)
            out_squeezed = self._se_reduce(out_squeezed)
            out_squeezed = self._swish(out_squeezed)
            out_squeezed = self._se_expand(out_squeezed)
            out = out * torch.sigmoid(out_squeezed)

        # Final phase
        out = self._project_conv(out)
        out = self._bn2(out)

        # Skip connection and Drop connect
        if self.id_skip:
            if self.training and drop_connect_rate is not None:
                out = drop_connect(out, drop_connect_rate)
            out += identity

        return out


@BACKBONES.register_module
class EfficientNet(nn.Module):
    """The class implements EfficientNet.
    """

    arch_types = {
        # width_coefficient, depth_coefficient, resolution, dropout_rate
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    }

    arch_settings = {
        # kernel_size, num_repeat, inplanes, outplanes, expansion,
        # id_skip, stride, se_ratio
        'stage-1': StageArgs(3, 1, 32, 16, 1, True, [1], 0.25),
        'stage-2': StageArgs(3, 2, 16, 24, 6, True, [2], 0.25),
        'stage-3': StageArgs(5, 2, 24, 40, 6, True, [2], 0.25),
        'stage-4': StageArgs(3, 3, 40, 80, 6, True, [2], 0.25),
        'stage-5': StageArgs(5, 3, 80, 112, 6, True, [1], 0.25),
        'stage-6': StageArgs(5, 4, 112, 192, 6, True, [2], 0.25),
        'stage-7': StageArgs(3, 1, 192, 320, 6, True, [1], 0.25)
    }

    bn_momentum = 0.99
    bn_eps = 1e-3
    drop_connect_rate = 0.2
    depth_divisor = 8
    min_depth = None
    use_se = True

    # TODO: Specify out_indices correctly.
    def __init__(self,
                 cls_name,
                 num_classes,
                 num_stages,
                 image_size,
                 out_indices=(0, 1, 2, 3),
                 style='pytorch'):
        super(EfficientNet, self).__init__()
        assert cls_name in self.arch_types, 'Wrong efficientnet configuration'
        assert len(self.arch_settings.keys()) == num_stages
        self.cls = self.arch_types[cls_name]
        self.width = self.cls[0]
        self.depth = self.cls[1]
        self.resolution = self.cls[2]
        self.dropout_rate = self.cls[3]
        self.num_classes = num_classes
        self.image_size = image_size
        self.style = style

        bn_momentum = 1 - self.bn_momentum

        Conv2d = get_same_padding_conv2d(self.image_size)
        outplanes = self.round_filters(self.arch_settings['stage-1'].inplanes)
        self._conv_stem = Conv2d(
            3,
            outplanes,
            3,
            stride=2,
            bias=False
        )

        self._bn0 = nn.BatchNorm2d(outplanes, self.bn_eps, bn_momentum)

        self._blocks = nn.ModuleList([])
        for stage in self.arch_settings.values():
            assert stage.num_repeat > 0
            num_repeat = self.round_repeats(stage.num_repeat)
            outplanes = self.round_filters(stage.outplanes)
            for i in range(num_repeat):
                # First block of a stage
                inplanes = self.round_filters(stage.inplanes)
                stride = stage.stride
                if i > 0:
                    inplanes = outplanes
                    stride = 1
                self._blocks.append(
                    MBConvBlock(
                        self.image_size,
                        inplanes,
                        outplanes,
                        stage.kernel_size,
                        stage.expansion,
                        stride,
                        self.bn_momentum,
                        self.bn_eps,
                        stage.se_ratio,
                        self.style
                    )
                )

        self._conv_head = Conv2d(
            outplanes,
            1280,
            1,
            bias=False
        )

        self._bn1 = nn.BatchNorm2d(1280, self.bn_eps, bn_momentum)
        # TODO: Remove the following comment
        self._global_pool = nn.AdaptiveAvgPool2d(1)
        self._fc = nn.Linear(1280, self.num_classes)
        if self.dropout_rate > 0:
            self._dropout = nn.Dropout2d(self.dropout_rate)
        self._swish = Swish()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                # elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                #     constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, MBConvBlock):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, MBConvBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def round_filters(self, filters):
        """Round number of filters based on depth multiplier.
        """
        orig_f = filters
        multiplier = self.width
        divisor = self.depth_divisor
        min_depth = self.min_depth
        if not multiplier:
            return filters

        filters *= multiplier
        min_depth = min_depth or divisor
        new_filters = max(min_depth,
                          int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        logging.info('round_filter input=%s output=%s', orig_f, new_filters)
        return int(new_filters)

    def round_repeats(self, repeats):
        """Round number of filters based on depth multiplier.
        """
        multiplier = self.depth
        if not multiplier:
            return repeats
        return int(math.ceil(multiplier * repeats))

    def forward(self, x):
        x = self._conv_stem(x)
        x = self._bn0(x)
        x = self._swish(x)
        outs = []
        for i, block in enumerate(self._blocks):
            x = block(x, self.drop_connect_rate)
            if i in self.out_indices:
                outs.append(x)

        x = self._conv_head(x)
        x = self._bn1(x)
        x = self._swish(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        if self.dropout_rate > 0:
            x = self._dropout(x)
        x = self._fc(x)

        return tuple(outs)
