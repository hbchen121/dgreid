import torch
import torch.nn as nn
import torch.nn.functional as F


def adv_loss(inputs, eps=1e-5):
    inputs = inputs.softmax(dim=1)
    loss = - torch.log(inputs + eps).mean(dim=1)
    return loss.mean()


class AdvLoss(torch.nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, inputs):
        return adv_loss(inputs, self.eps)


