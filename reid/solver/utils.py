# encoding: utf-8
from bisect import bisect_right
import torch


def get_lr(optimizer: torch.optim.Optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
