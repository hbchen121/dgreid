import torch
import torch.nn.functional as F
from torch.nn import init
from torch import nn, autograd
import numpy as np
from ..loss import adv_loss
from .memory import MemoryClassifier, mc


class DomainMemoryClassifier(nn.Module):
    def __init__(self, num_features, num_domains, temp=0.05, momentum=0.2):
        super(DomainMemoryClassifier, self).__init__()
        self.num_features = num_features
        self.num_domains = num_domains
        self.momentum = momentum
        self.temp = temp

        self.register_buffer('features', torch.zeros(num_domains, num_features))
        self.register_buffer('labels', torch.zeros(num_domains).long())

    def init(self, features, labels):
        """
        f: pid * dim, labels: pid * 1
        """
        self.features = features
        self.labels = labels

    def MomentumUpdate(self, inputs, indexes, mean_update=False):
        """
        inputs: batch(per_batch * num_domains) * dim, indexes: (per_batch * num_domains) * 1
        """
        # momentum update
        if mean_update:
            batch_size = inputs.shape[0]
            per_batch = batch_size // self.num_domains
            inputs = torch.stack(torch.chunk(inputs,  self.num_domains, dim=0), dim=0)  # per_batch * num_domains * dim
            inputs = inputs.mean(dim=0)  # num_domain * dim
            idx = torch.arange(0, self.num_domains, per_batch)
            indexes = indexes[idx]
        for x, y in zip(inputs, indexes):
            self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x
            self.features[y] = self.features[y] / self.features[y].norm()

    def forward(self, inputs, targets, adv=False):

        logits = mc(inputs, targets, self.features, self.momentum)  # B * C
        logits = logits / self.temp

        if adv:
            loss = adv_loss(logits)
        else:
            loss = F.cross_entropy(logits, targets)
        return loss



