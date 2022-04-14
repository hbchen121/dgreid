from __future__ import absolute_import

from .resnet import *
from .resnet_plus import *
from .idm_module import *
from .resnet_idm import *
from .resnet_ibn import *
from .resnet_ibn_idm import *
from .resnet_mde import *
from .IBNMeta import MetaIBNet
from .resMeta import MetaResNet
from .resMetaMix import MetaResNetMix
from .resnet_mde_v2 import resnet50_mde_v2
from .resnet_meta_v2 import resnet50_meta_v2


__factory = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet50_plus': resnet50_plus,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnet_ibn50a': resnet_ibn50a,
    'resnet_ibn101a': resnet_ibn101a,
    'resnet50_idm': resnet50_idm,
    'resnet_ibn50a_idm': resnet_ibn50a_idm,
    'resnet50_mde': resnet50_mde,
    'resnet50_mde_v2': resnet50_mde_v2,  # batch data with arrange
    'resnet50_meta_v2': resnet50_meta_v2,  # batch data with arrange
    'resMeta': MetaResNet,
    'resMetaMix': MetaResNetMix,
    'IBNMeta': MetaIBNet,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a model instance.

    Parameters
    ----------
    name : str
        Model name. Can be one of 'inception', 'resnet18', 'resnet34',
        'resnet50', 'resnet101', and 'resnet152'.
    pretrained : bool, optional
        Only applied for 'resnet*' models. If True, will use ImageNet pretrained
        model. Default: True
    cut_at_pooling : bool, optional
        If True, will cut the model before the last global pooling layer and
        ignore the remaining kwargs. Default: False
    num_features : int, optional
        If positive, will append a Linear layer after the global pooling layer,
        with this number of output units, followed by a BatchNorm layer.
        Otherwise these layers will not be appended. Default: 256 for
        'inception', 0 for 'resnet*'
    norm : bool, optional
        If True, will normalize the feature to be unit L2-norm for each sample.
        Otherwise will append a ReLU layer after the above Linear layer if
        num_features > 0. Default: False
    dropout : float, optional
        If positive, will append a Dropout layer with this dropout rate.
        Default: 0
    num_classes : int, optional
        If positive, will append a Linear layer at the end as the classifier
        with this number of output units. Default: 0
    """
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)
