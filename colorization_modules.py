import logging as log
import os
from pathlib import Path

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch import nn
from torch.nn import functional as F

from enum import Enum


class AdvEnum(Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

    @classmethod
    def list_name_value(cls):
        return list(map(lambda c: (c.name, c.value), cls))


class DecoNetMode(AdvEnum):
    FREEZE_DECO = 0
    FREEZE_PTMODEL = 1
    FREEZE_PTMODEL_NO_FC = 2
    UNFREEZE_ALL = 3
    FREEZE_ALL = 4
    FREEZE_ALL_NO_FC = 5


class DecoType(AdvEnum):
    NO = 0
    DECONV = 1
    RESIZE_CONV = 2
    ColorUDECO = 16
    PIXEL_SHUFFLE = 20


def get_deco_model(use_deco, out_deco) -> nn.Module:
    if use_deco in [DecoType.DECONV, DecoType.DECONV_NORM]:
        return StandardDECO(out_deco, deconv=True)
    elif use_deco in [DecoType.RESIZE_CONV]:
        return StandardDECO(out_deco, deconv=False)
    elif use_deco is DecoType.PIXEL_SHUFFLE:
        return PixelShuffle(out_deco, lrelu=False)
    elif use_deco is DecoType.ColorUDECO:
        return ColorUDECO(out_deco)
    else:
        raise ValueError("Module not found")


class PreTrainedModel(AdvEnum):
    DENSENET_121 = 0
    RESNET_18 = 1
    RESNET_34 = 2
    RESNET_50 = 3
    VGG11 = 4
    VGG11_BN = 5


def get_pt_model(model, output, pretrained=True):
    input = 224
    if not isinstance(model, PreTrainedModel):
        model = PreTrainedModel(model)
    pt_model = None
    if model == PreTrainedModel.DENSENET_121:
        pt_model = models.densenet121(pretrained=pretrained)
        num_ftrs = pt_model.classifier.in_features
        pt_model.classifier = nn.Linear(num_ftrs, output)
        pt_model.last_layer_name = "classifier"
    elif model == PreTrainedModel.RESNET_18:
        pt_model = models.resnet18(pretrained=pretrained)
        num_ftrs = pt_model.fc.in_features
        pt_model.fc = nn.Linear(num_ftrs, output)
        pt_model.last_layer_name = "fc"
    elif model == PreTrainedModel.RESNET_34:
        pt_model = models.resnet34(pretrained=pretrained)
        num_ftrs = pt_model.fc.in_features
        pt_model.fc = nn.Linear(num_ftrs, output)
        pt_model.last_layer_name = "fc"
    elif model == PreTrainedModel.RESNET_50:
        pt_model = models.resnet50(pretrained=pretrained)
        num_ftrs = pt_model.fc.in_features
        pt_model.fc = nn.Linear(num_ftrs, output)
        pt_model.last_layer_name = "fc"
    elif model == PreTrainedModel.VGG11:
        pt_model = models.vgg11(pretrained=pretrained)
        num_ftrs = pt_model.classifier[6].in_features
        pt_model.classifier[6] = nn.Linear(num_ftrs, output)
        pt_model.last_layer_name = "classifier.6"
    elif model == PreTrainedModel.VGG11_BN:
        pt_model = models.vgg11_bn(pretrained=pretrained)
        num_ftrs = pt_model.classifier[6].in_features
        pt_model.classifier[6] = nn.Linear(num_ftrs, output)
        pt_model.last_layer_name = "classifier.6"
    else:
        raise ValueError("Model not found")

    return pt_model, input


class DecoNet(nn.Module):
    """
    Colorization module(optional)+Model
    """

    def __init__(self, output=14,
                 deco_type=DecoType.ColorUDECO,
                 pt_model=PreTrainedModel.RESNET_18,
                 pre_trained=True,
                 training_mode=DecoNetMode.FREEZE_PTMODEL_NO_FC,
                 use_aap=False):
        super().__init__()
        # Pre-trained Model
        self.deco_type = deco_type
        self.training_mode = training_mode
        self.use_aap = use_aap
        pt_model, self.out_deco = get_pt_model(pt_model, output, pre_trained)
        self.last_layer_name = pt_model.last_layer_name
        # DECO if needed
        if self.deco_type is not DecoType.NO:
            self.deco = get_deco_model(self.deco_type, self.out_deco)
        else:
            self.deco = None
        self.pt_model = pt_model
        self.set_mode(training_mode)

    def set_mode(self, mode, print=True):
        if not isinstance(mode, DecoNetMode):
            mode = DecoNetMode(mode)
        if mode == DecoNetMode.UNFREEZE_ALL:
            for param in self.parameters():
                param.requires_grad = True
        elif mode == DecoNetMode.FREEZE_DECO:
            self.set_mode(DecoNetMode.UNFREEZE_ALL, False)
            for param in self.deco.parameters():
                param.requires_grad = False
        elif mode == DecoNetMode.FREEZE_PTMODEL:
            self.set_mode(DecoNetMode.UNFREEZE_ALL, False)
            for param in self.pt_model.parameters():
                param.requires_grad = False
        elif mode == DecoNetMode.FREEZE_PTMODEL_NO_FC:
            self.set_mode(DecoNetMode.UNFREEZE_ALL, False)
            for name, param in self.pt_model.named_parameters():
                if self.last_layer_name not in name:
                    param.requires_grad = False
        elif mode == DecoNetMode.FREEZE_ALL:
            for param in self.parameters():
                param.requires_grad = False
        elif mode == DecoNetMode.FREEZE_ALL_NO_FC:
            self.set_mode(DecoNetMode.FREEZE_ALL, False)
            # Unfreeze last layer
            for name, param in self.pt_model.named_parameters():
                if self.last_layer_name in name:
                    param.requires_grad = True

        if print:
            log.info("#############################################")
            log.info("PARAMETERS STATUS:")
            for name, param in self.named_parameters():
                log.info("{} : {}".format(name, param.requires_grad))
            log.info("#############################################")

    def get_layer_weight(self, sel_name: str = ""):
        if sel_name == "":
            sel_name = self.last_layer_name
        res = []
        for name, param in self.pt_model.named_parameters():
            if sel_name in name:
                res.append(param)

        return res

    def forward(self, xb):
        """
        @:param xb : tensor
          Batch of input images

        @:return tensor
          A batch of output images
        """
        if self.deco is not None:
            xb = self.deco(xb)
            if self.use_aap:
                xb = F.adaptive_avg_pool2d(xb, (self.out_deco, self.out_deco))
        return self.pt_model(xb)

    def clean_last_layer(self):
        pt_model_type = self.pt_model

        if pt_model_type == PreTrainedModel.VGG11_BN or pt_model_type == PreTrainedModel.VGG11:
            self.pt_model.classifier[6].reset_parameters()
        else:
            last_layer_name = list(self.pt_model._modules)[-1]
            self.pt_model._modules[last_layer_name].reset_parameters()

        log.info("Last layer cleaned!")

    def last_layer_size(self):
        pt_model_type = self.pt_model
        if pt_model_type == PreTrainedModel.VGG11_BN or pt_model_type == PreTrainedModel.VGG11:
            return self.pt_model.classifier[6].weight.shape[-1]
        else:
            last_layer_name = list(self.pt_model._modules)[-1]
            return self.pt_model._modules[last_layer_name].shape[-1]

    def load_deco_state_dict(self, state_dict):
        if self.deco is None:
            self.deco = get_deco_model(self.deco_type, self.out_deco)
        if hasattr(self.deco, "load_state_dict"):
            self.deco.load_state_dict(state_dict)
        else:
            return False
        self.set_mode(self.training_mode)
        return True


def default_deco__weight_init(m):
    if isinstance(m, nn.Conv2d):
        # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        # m.weight.data.normal_(0, math.sqrt(2. / n))
        torch.nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def bn_weight_init(m):
    if isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


class BaseDECO(nn.Module):
    def __init__(self, out=224, init=None):
        super().__init__()
        self.out_s = out
        self.init = init

    def set_output_size(self, out_s):
        self.out_s = out_s

    def init_weights(self):
        if self.init is None:
            pass
        elif self.init == 0:
            self.apply(default_deco__weight_init)
        elif self.init == 1:
            self.apply(bn_weight_init)


class ResBlock(nn.Module):
    def __init__(self, ni, nf=None, kernel=3, stride=1, padding=1):
        super().__init__()
        if nf is None:
            nf = ni
        self.conv1 = conv_layer(ni, nf, kernel=kernel, stride=stride, padding=padding)
        self.conv2 = conv_layer(nf, nf, kernel=kernel, stride=stride, padding=padding)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


def conv_layer(in_layer, out_layer, kernel=3, stride=1, padding=1, instanceNorm=False):
    return nn.Sequential(
        nn.Conv2d(in_layer, out_layer, kernel_size=kernel, stride=stride, padding=padding),
        nn.BatchNorm2d(out_layer) if not instanceNorm else nn.InstanceNorm2d(out_layer),
        nn.LeakyReLU(inplace=True)
    )


def _make_res_layers(nl, ni, kernel=3, stride=1, padding=1):
    layers = []
    for i in range(nl):
        layers.append(ResBlock(ni, kernel=kernel, stride=stride, padding=padding))

    return nn.Sequential(*layers)


class StandardDECO(BaseDECO):
    """
    Standard DECO Module
    """

    def __init__(self, out=224, init=0, deconv=False):
        super().__init__(out, init)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        # ReLU
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks = _make_res_layers(8, 64)
        self.conv_last = nn.Conv2d(64, 3, kernel_size=1)
        self.deconv = deconv
        if deconv:
            # TODO: Check if use "groups = 1"
            self.deconv = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=8, padding=2, stride=4,
                                             groups=3, bias=False)
        else:
            self.pad = nn.ReflectionPad2d(1)
            self.conv_up = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=0, stride=1)

        self.init_weights()

    def forward(self, xb):
        """
        @:param xb : Tensor
          Batch of input images

        @:return tensor
          A batch of output images
        """
        _xb = self.maxpool(F.leaky_relu(self.bn1(self.conv1(xb))))
        _xb = self.resblocks(_xb)
        _xb = self.conv_last(_xb)
        if self.deconv:
            _xb = self.deconv(_xb, output_size=xb.shape)
        else:
            _xb = self.conv_up(self.pad(F.interpolate(_xb, scale_factor=4, mode='nearest')))
        return _xb


def icnr(x, scale=4, init=nn.init.kaiming_normal_):
    """ ICNR init of `x`, with `scale` and `init` function.

        Checkerboard artifact free sub-pixel convolution: https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """
    ni, nf, h, w = x.shape
    ni2 = int(ni / (scale ** 2))
    k = init(torch.zeros([ni2, nf, h, w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale ** 2)
    k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
    x.data.copy_(k)


class PixelShuffle_ICNR(nn.Module):
    """ Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`, `icnr` init,
        and `weight_norm`.

        "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts":
        https://arxiv.org/abs/1806.02658
    """

    def __init__(self, ni: int, nf: int = None, scale: int = 4, icnr_init=True, blur_k=2, blur_s=1,
                 blur_pad=(1, 0, 1, 0), lrelu=True):
        super().__init__()
        nf = ni if nf is None else nf
        self.conv = conv_layer(ni, nf * (scale ** 2), kernel=1, padding=0, stride=1) if lrelu else nn.Sequential(
            nn.Conv2d(64, 3 * (scale ** 2), 1, 1, 0), nn.BatchNorm2d(3 * (scale ** 2)))
        if icnr_init:
            icnr(self.conv[0].weight, scale=scale)
        self.act = nn.LeakyReLU(inplace=False) if lrelu else nn.Hardtanh(-10000, 10000)
        self.shuf = nn.PixelShuffle(scale)
        # Blurring over (h*w) kernel
        self.pad = nn.ReplicationPad2d(blur_pad)
        self.blur = nn.AvgPool2d(blur_k, stride=blur_s)

    def forward(self, x):
        x = self.shuf(self.act(self.conv(x)))
        return self.blur(self.pad(x))


class PixelShuffle(BaseDECO):
    """
    PixelShuffle Module
    """

    def __init__(self, out=224, init=1, scale=4, lrelu=False):
        super().__init__(out, init)
        # Which value should I use for stride and padding?
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.LeakyReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks = _make_res_layers(8, 64)
        self.pixel_shuffle = PixelShuffle_ICNR(ni=64, nf=3, scale=scale, lrelu=lrelu)
        self.init_weights()

    def forward(self, xb):
        """
        @:param xb : Tensor
          Batch of input images

        @:return tensor
          A batch of output images
        """
        _xb = self.maxpool(self.act1(self.bn1(self.conv1(xb))))
        _xb = self.resblocks(_xb)

        return self.pixel_shuffle(_xb)


class ColorUDECO(BaseDECO):
    """
    ColorUDECO Module
    """

    def __init__(self, out=224, init=0, in_ch=1, out_ch=3):
        super().__init__(out, init)
        self.dw1 = ColorDown(in_ch, 16)
        self.dw2 = ColorDown(16, 32)
        self.dw3 = ColorDown(32, 64)
        self.up1 = ColorUp(64, 32)
        self.up2 = ColorUp(64, 16)
        self.out = ColorOut(32, 16, out_ch)

    def forward(self, x1):
        """
        @:param x1 : Tensor
          Batch of input images

        @:return tensor
          A batch of output images
        """
        x1 = self.dw1(x1)
        x2 = self.dw2(x1)
        x3 = self.dw3(x2)
        x3 = self.up1(x3)
        x2 = self.up2(torch.cat([x2, x3], dim=1))
        return self.out(torch.cat([x1, x2], dim=1))


class ColorDown(nn.Module):
    def __init__(self, in_ch, out_ch, htan=False):
        super(ColorDown, self).__init__()
        self.d = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU() if not htan else nn.Hardtanh(),
            nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU() if not htan else nn.Hardtanh(),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        return self.d(x)


class ColorUp(nn.Module):
    def __init__(self, in_ch, out_ch, htan=False):
        super(ColorUp, self).__init__()
        self.u = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
            nn.LeakyReLU() if not htan else nn.Hardtanh(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU() if not htan else nn.Hardtanh(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU() if not htan else nn.Hardtanh(),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        return self.u(x)


class ColorOut(nn.Module):
    def __init__(self, in_ch, out_ch, out_last, htan=False):
        super(ColorOut, self).__init__()
        self.u = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
            nn.LeakyReLU() if not htan else nn.Hardtanh(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU() if not htan else nn.Hardtanh(),
            nn.Conv2d(out_ch, out_last, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.u(x)
