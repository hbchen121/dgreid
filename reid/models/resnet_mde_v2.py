from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
from collections import OrderedDict
import math
from reid.layers import (
    IBN,
    SELayer,
    Non_local,
    get_norm,
    SRMLayer,
    CSRMLayer,
    SMMLayer,
)
from reid.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message


__all__ = [
    'resnet50_mde_v2',
]

model_urls = {
    '18x': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    '34x': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    '50x': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    '101x': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'ibn_18x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet18_ibn_a-2f571257.pth',
    'ibn_34x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet34_ibn_a-94bc1577.pth',
    'ibn_50x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth',
    'ibn_101x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_a-59ea0ac6.pth',
    'se_ibn_101x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/se_resnet101_ibn_a-fabed4e2.pth',
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, bn_norm, with_ibn=False, with_se=False, with_in=False,
                 with_srm=False, with_csrm=False, stride=1, downsample=None, reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if with_ibn:
            self.bn1 = IBN(planes, bn_norm)
        else:
            self.bn1 = get_norm(bn_norm, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = get_norm(bn_norm, planes)
        self.relu = nn.ReLU(inplace=True)
        if with_se:
            self.se = SELayer(planes, reduction)
        else:
            self.se = nn.Identity()
        if with_srm:
            self.srm = SRMLayer(planes)
        else:
            self.srm = nn.Identity()
        if with_csrm:
            self.csrm = CSRMLayer(planes, track_running_stats=True)
        else:
            self.csrm = nn.Identity()
        if with_in:
            self.IN = nn.InstanceNorm2d(planes, affine=True)
        else:
            self.IN = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out = self.srm(out)
        out = self.csrm(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        out = self.IN(out)

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, bn_norm, with_ibn=False, with_se=False, with_in=False,
                 with_srm=False, with_csrm=False, stride=1, downsample=None, reduction=16):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if with_ibn:
            self.bn1 = IBN(planes, bn_norm)
        else:
            self.bn1 = get_norm(bn_norm, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = get_norm(bn_norm, planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = get_norm(bn_norm, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        if with_se:
            self.se = SELayer(planes * self.expansion, reduction)
        else:
            self.se = nn.Identity()
        if with_srm:
            self.srm = SRMLayer(planes * self.expansion)
        else:
            self.srm = nn.Identity()
        if with_csrm:
            self.csrm = CSRMLayer(planes * self.expansion, track_running_stats=False)
        else:
            self.csrm = nn.Identity()
        # IN following DualNorm position
        if with_in and False:
            self.IN = nn.InstanceNorm2d(planes * self.expansion, affine=True)
        else:
            self.IN = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)
        out = self.srm(out)
        out = self.csrm(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.IN(out)

        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, last_stride, bn_norm, with_ibn, with_se, with_nl, with_in,
                 with_srm, with_csrm, with_smm, block, layers, non_layers, pretrained,
                 cut_at_pooling, num_features, norm, dropout, num_classes, args):
        self.inplanes = 64
        super().__init__()

        self.n_source = len(num_classes)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = get_norm(bn_norm, 64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0], 1, bn_norm, with_ibn, with_se, with_in, with_srm, with_csrm)
        self.layer2 = self._make_layer(block, 128, layers[1], 2, bn_norm, with_ibn, with_se, with_in, with_srm, with_csrm)
        self.layer3 = self._make_layer(block, 256, layers[2], 2, bn_norm, with_ibn, with_se, with_in, with_srm, with_csrm)
        self.layer4 = self._make_layer(block, 512, layers[3], last_stride, bn_norm,
                                       with_se=with_se, with_srm=with_srm, with_csrm=with_csrm)

        # self.random_init()
        if not pretrained:
            self.reset_params()

        # fmt: off
        if with_nl: self._build_nonlocal(layers, non_layers, bn_norm)
        else:       self.NL_1_idx = self.NL_2_idx = self.NL_3_idx = self.NL_4_idx = []
        # fmt: on

        # IN following SNR
        if with_in:
            self.IN1 = nn.InstanceNorm2d(64 * 4, affine=True)
            self.IN2 = nn.InstanceNorm2d(128 * 4, affine=True)
            self.IN3 = nn.InstanceNorm2d(256 * 4, affine=True)
            # self.IN4 = nn.InstanceNorm2d(512, affine=True)
        else:
            self.IN1 = self.IN2 = self.IN3 = nn.Identity()
            # self.IN1 = self.IN2 = self.IN3 = self.IN4 = nn.Identity()

        with_smm, smm_stage, smm_half, smm_in, smm_lam, \
        smm_half_id, num_instances, mix_layer, smm_scale = with_smm
        self.with_smm_loss = mix_layer != 'AdaIN'
        self.smm_scale = smm_scale
        self.with_smm = with_smm
        if with_smm:
            dims = [3, 64, 256, 512, 1024, 2048]
            self.smm_stage = smm_stage
            self.smm = nn.ModuleList()
            j = 0
            for stage, dim in enumerate(dims):
                if (stage - 1) in self.smm_stage:
                    smm = SMMLayer(dim, self.n_source, smm_half, smm_in,
                                   smm_lam, smm_half_id, num_instances,
                                   mix_layer, smm_scale[j])
                    j += 1
                else:
                    smm = nn.Identity()
                self.smm.append(smm)

        # heads
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.cut_at_pooling = cut_at_pooling
        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes
            self.num_classes_all = 0

            out_planes = 2048

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features, bias=False)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                # self.feat_bn = get_norm(bn_norm, self.num_features)
                nn.init.normal_(self.feat.weight, 0, 0.01)
                # init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            self.feat_bn.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)

            if self.num_classes[0] != 0:
                self.num_classes_all = sum(num_classes)
                if args is not None and args.local_classifiers:
                    self.classifiers = nn.ModuleList()
                    for num_class in self.num_classes:
                        classifier = nn.Linear(self.num_features, num_class, bias=False)
                        init.normal_(classifier.weight, std=0.001)
                        self.classifiers.append(classifier)
                else:
                    self.classifiers = None
                if args is not None and args.global_classifier:
                    self.global_classifier = nn.Linear(self.num_features, self.num_classes_all, bias=False)
                    init.normal_(self.global_classifier.weight, std=0.001)
                else:
                    self.global_classifier = None

        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, bn_norm="BN",
                    with_ibn=False, with_se=False, with_in=False, with_srm=False, with_csrm=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                get_norm(bn_norm, planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, bn_norm, with_ibn, with_se, with_in, with_srm, with_csrm, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_norm, with_ibn, with_se, with_in, with_srm, with_csrm))

        return nn.Sequential(*layers)

    def _build_nonlocal(self, layers, non_layers, bn_norm):
        self.NL_1 = nn.ModuleList(
            [Non_local(256, bn_norm) for _ in range(non_layers[0])])
        self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
        self.NL_2 = nn.ModuleList(
            [Non_local(512, bn_norm) for _ in range(non_layers[1])])
        self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
        self.NL_3 = nn.ModuleList(
            [Non_local(1024, bn_norm) for _ in range(non_layers[2])])
        self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
        self.NL_4 = nn.ModuleList(
            [Non_local(2048, bn_norm) for _ in range(non_layers[3])])
        self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

    def backbone_forward(self, x, identity=False):
        if self.with_smm and -1 in self.smm_stage:
            if self.with_smm_loss and self.training:
                x, mix_mean, mix_std, style_mean, style_std = self.smm[0](x, identity)
            else:
                x = self.smm[0](x, identity)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if self.with_smm and 0 in self.smm_stage:
            if self.with_smm_loss and self.training:
                x, mix_mean, mix_std, style_mean, style_std = self.smm[1](x, identity)
            else:
                x = self.smm[1](x, identity)

        # layer 1
        NL1_counter = 0
        if len(self.NL_1_idx) == 0:
            self.NL_1_idx = [-1]
        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
            if i == self.NL_1_idx[NL1_counter]:
                _, C, H, W = x.shape
                x = self.NL_1[NL1_counter](x)
                NL1_counter += 1
        x = self.IN1(x)
        if self.with_smm and 1 in self.smm_stage:
            if self.with_smm_loss and self.training:
                x, mix_mean, mix_std, style_mean, style_std = self.smm[2](x, identity)
            else:
                x = self.smm[2](x, identity)

        # layer 2
        NL2_counter = 0
        if len(self.NL_2_idx) == 0:
            self.NL_2_idx = [-1]
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
            if i == self.NL_2_idx[NL2_counter]:
                _, C, H, W = x.shape
                x = self.NL_2[NL2_counter](x)
                NL2_counter += 1
        x = self.IN2(x)
        if self.with_smm and 2 in self.smm_stage:
            if self.with_smm_loss and self.training:
                x, mix_mean, mix_std, style_mean, style_std = self.smm[3](x, identity)
            else:
                x = self.smm[3](x, identity)

        # layer 3
        NL3_counter = 0
        if len(self.NL_3_idx) == 0:
            self.NL_3_idx = [-1]
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)
            if i == self.NL_3_idx[NL3_counter]:
                _, C, H, W = x.shape
                x = self.NL_3[NL3_counter](x)
                NL3_counter += 1
        x = self.IN3(x)
        if self.with_smm and 3 in self.smm_stage:
            if self.with_smm_loss and self.training:
                x, mix_mean, mix_std, style_mean, style_std = self.smm[4](x, identity)
            else:
                x = self.smm[4](x, identity)

        # layer 4
        NL4_counter = 0
        if len(self.NL_4_idx) == 0:
            self.NL_4_idx = [-1]
        for i in range(len(self.layer4)):
            x = self.layer4[i](x)
            if i == self.NL_4_idx[NL4_counter]:
                _, C, H, W = x.shape
                x = self.NL_4[NL4_counter](x)
                NL4_counter += 1
        if self.with_smm and 4 in self.smm_stage:
            if self.with_smm_loss and self.training:
                x, mix_mean, mix_std, style_mean, style_std = self.smm[5](x, identity)
            else:
                x = self.smm[5](x, identity)
        if self.with_smm and self.with_smm_loss and self.training:
            return x, mix_mean, mix_std, style_mean, style_std
        return x

    def heads_forward(self, x):
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        if self.cut_at_pooling:
            return x

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

        if (self.training is False):
            # bn_x = F.normalize(bn_x)
            return bn_x

        if self.norm:
            norm_bn_x = F.normalize(bn_x)
        elif self.has_embedding:
            bn_x = F.relu(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        probs = []
        g_probs = []
        if self.num_classes[0] != 0:
            bn_x = torch.chunk(bn_x, self.n_source, dim=0)
            for i in range(self.n_source):
                if self.classifiers:
                    prob = self.classifiers[i](bn_x[i])
                    probs.append(prob)
                if self.global_classifier:
                    global_prob = self.global_classifier(bn_x[i])
                    g_probs.append(global_prob)
            # probs = torch.cat(tuple(probs), dim=0)


        # else:
        #     return bn_x
        x_n = torch.chunk(x, self.n_source, dim=0)

        if self.norm:
            norm_x_n = torch.chunk(norm_bn_x, self.n_source, dim=0)
            return probs, g_probs, x_n, norm_x_n
        else:
            return probs, g_probs, x_n

    def forward(self, x, identity=False):
        if self.with_smm_loss and self.training:
            x, mix_mean, mix_std, style_mean, style_std = self.backbone_forward(x, identity)
            x = self.heads_forward(x)
            return x, mix_mean, mix_std, style_mean, style_std
        x = self.backbone_forward(x, identity)
        x = self.heads_forward(x)
        return x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


def init_pretrained_weights(key):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    import os
    import errno
    import gdown

    def _get_torch_home():
        ENV_TORCH_HOME = 'TORCH_HOME'
        ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
        DEFAULT_CACHE_DIR = '~/.cache'
        torch_home = os.path.expanduser(
            os.getenv(
                ENV_TORCH_HOME,
                os.path.join(
                    os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch'
                )
            )
        )
        return torch_home

    torch_home = _get_torch_home()
    model_dir = os.path.join(torch_home, 'checkpoints')
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    filename = model_urls[key].split('/')[-1]

    cached_file = os.path.join(model_dir, filename)

    if not os.path.exists(cached_file):
        print(f"Pretrain model don't exist, downloading from {model_urls[key]}")
        gdown.download(model_urls[key], cached_file, quiet=False)


    print(f"Loading pretrained model from {cached_file}")
    state_dict = torch.load(cached_file, map_location=torch.device('cpu'))

    return state_dict


def resnet_mde_v2(depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=None, args=None):

    # fmt: off
    pretrain      = pretrained
    pretrain_path = ''
    last_stride   = 1
    bn_norm       = args.bn_type
    with_ibn      = args.with_ibn
    with_se       = args.with_se
    with_nl       = args.with_nl
    with_in       = args.with_in
    with_srm      = args.with_srm
    with_csrm     = args.with_csrm
    with_smm      = [args.with_smm, args.smm_stage, args.smm_half, args.smm_in,
                     args.smm_lam, args.smm_half_id, args.num_instances,
                     args.smm_mix_layer, args.smm_scale]
    depth         = depth
    # fmt: on

    num_blocks_per_stage = {
        '18x': [2, 2, 2, 2],
        '34x': [3, 4, 6, 3],
        '50x': [3, 4, 6, 3],
        '101x': [3, 4, 23, 3],
    }[depth]

    nl_layers_per_stage = {
        '18x': [0, 0, 0, 0],
        '34x': [0, 0, 0, 0],
        '50x': [0, 2, 3, 0],
        '101x': [0, 2, 9, 0]
    }[depth]

    block = {
        '18x': BasicBlock,
        '34x': BasicBlock,
        '50x': Bottleneck,
        '101x': Bottleneck
    }[depth]

    model = ResNet(last_stride, bn_norm, with_ibn, with_se, with_nl, with_in,
                   with_srm, with_csrm, with_smm,
                   block, num_blocks_per_stage, nl_layers_per_stage, pretrained,
                   cut_at_pooling, num_features, norm, dropout, num_classes, args)
    if pretrain:
        # Load pretrain path if specifically
        if pretrain_path:
            try:
                state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
                print(f"Loading pretrained model from {pretrain_path}")
            except FileNotFoundError as e:
                print(f'{pretrain_path} is not found! Please check this path.')
                raise e
            except KeyError as e:
                print("State dict keys error! Please check the state dict.")
                raise e
        else:
            key = depth
            if with_ibn: key = 'ibn_' + key
            if with_se:  key = 'se_' + key

            state_dict = init_pretrained_weights(key)

        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            print(
                get_missing_parameters_message(incompatible.missing_keys)
            )
        if incompatible.unexpected_keys:
            print(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )

    return model


def resnet50_mde_v2(**kwargs):
    return resnet_mde_v2('50x', **kwargs)
