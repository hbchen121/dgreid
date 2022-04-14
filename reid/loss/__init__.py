from __future__ import absolute_import

from .triplet import TripletLoss
from .triplet_xbm import TripletLossXBM
from .crossentropy import CrossEntropyLabelSmooth, CrossEntropy
from .idm_loss import DivLoss, BridgeFeatLoss, BridgeProbLoss
from .adv_loss import *

# __all__ = [
#     'DivLoss',
#     'BridgeFeatLoss',
#     'BridgeProbLoss',
#     'TripletLoss',
#     'TripletLossXBM',
#     'CrossEntropyLabelSmooth',
#     'CrossEntropy',
# ]
