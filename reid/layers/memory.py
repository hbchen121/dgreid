import torch
import torch.nn.functional as F
from torch.nn import init
from torch import nn, autograd
import numpy as np
from reid.loss.crossentropy import CrossEntropy
from .softmaxs import cosSoftmax, arcSoftmax, circleSoftmax
import math

__all__ = [
    "MemoryClassifier",
    "MemoryClassifierPairs"
]

class MC(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        return grad_inputs, None, None, None


def mc(inputs, indexes, features, momentum=0.5):
    return MC.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class MemoryClassifier(nn.Module):
    def __init__(self, num_features, num_samples, mem_type='cos', temp=0.05, momentum=0.2, margin=0., num_instances=4,
                 dynamic_momentum=0, decay=0.9999):
        super(MemoryClassifier, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.type = mem_type
        self.margin = margin
        self.temp = temp
        self.num_instances = num_instances
        if dynamic_momentum > 0:
            self.decay = lambda x, m: decay * (1 - math.exp(-x / dynamic_momentum))
        else:
            self.decay = lambda x, m: m
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('labels', torch.zeros(num_samples).long())

    def init(self, features, labels):
        """
        f: pid * dim, labels: pid * 1
        """
        self.features = features
        self.labels = labels

    def MomentumUpdate(self, inputs, indexes, mean_update=False):
        # momentum update
        if mean_update:
            batch_size = inputs.shape[0]
            n_pid = batch_size // self.num_instances
            inputs = torch.stack(torch.chunk(inputs, n_pid, dim=0), dim=0)  # pid * intances * dim
            inputs = inputs.mean(dim=1)  # pid * dim
            pids_mean_idx = torch.arange(0, batch_size, self.num_instances)
            indexes = indexes[pids_mean_idx]
        for x, y in zip(inputs, indexes):
            self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x
            self.features[y] = self.features[y] / self.features[y].norm()

    def forward(self, inputs, targets, iter=0):

        self.momentum = self.decay(iter, self.momentum)
        logits = mc(inputs, targets, self.features, self.momentum)  # B * C

        if self.type == 'cos':
            if self.margin > 0:
                logits = cosSoftmax(logits, targets, self.margin)
        elif self.type == 'arc':
            logits = arcSoftmax(logits, targets, self.margin)
        elif self.type == 'circle':
            logits = circleSoftmax(logits, targets, self.margin)
        else:
            assert False, "invalid type {}".format(self.type)

        logits = logits / self.temp
        
        loss = F.cross_entropy(logits, targets)
        return loss

    def forward_back(self, inputs, indexes):

        sim = mc(inputs, indexes, self.features, self.momentum)  ## B * C

        sim = sim / self.temp

        loss = F.cross_entropy(sim, indexes)
        return loss


class MemoryClassifierPairs(MemoryClassifier):
    def __init__(self, num_features, num_samples, mem_type='cos', temp=0.05, momentum=0.2, margin=0., num_instances=4):
        super(MemoryClassifierPairs, self).__init__(num_features, num_samples, mem_type, temp, momentum, margin, num_instances)
        self.features = torch.zeros(num_samples, num_instances, num_features)

    def repeat_instances(self, input, dim=1):
        """
        input: pid * dim
        return: pid * num_instance * dim
        """
        input = input.unsqueeze(dim=dim)  # pid * 1 * dim
        shape = torch.ones_like(torch.tensor(input.size()))
        shape[dim] = self.num_instances
        input = input.repeat(*shape)  # pid * num_instance * dim
        return input

    def init(self, features, labels):
        """
        f: pid * dim, labels: pid * 1
        """
        self.features = self.repeat_instances(features)
        self.labels = labels

    def MomentumUpdate(self, inputs, indexes, mean_update=False):
        """
        inputs: (n_pid * ins) * dim, indexes: (n_pid * ins) * 1
        """
        batch_size = inputs.shape[0]
        n_pid = batch_size // self.num_instances
        inputs = torch.chunk(inputs, n_pid, dim=0)  # n_pid * (ins * dim)
        pids_mean_idx = torch.arange(0, batch_size, self.num_instances)
        indexes = indexes[pids_mean_idx]  # n_pid * 1
        for x, y in zip(inputs, indexes):
            self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x
            self.features[y] = F.normalize(self.features[y], dim=-1)
            # self.features[y] = self.features[y] / self.features[y].norm()

    def forward(self, inputs, targets):
        """
        input: batch * dim, targets: batch * 1
        """

        batch, dim = inputs.shape

        # targets = self.repeat_instances(targets)

        features = self.features.view(-1, dim)  # (pid * num_instance) * dim
        dist_mat = torch.matmul(inputs, features.t())  # batch * (pid * ins)

        index = torch.where(targets != -1)[0]
        one_hot = torch.zeros(index.size()[0], self.features.size()[0], device=dist_mat.device, dtype=torch.long)
        one_hot.scatter_(1, targets[index, None], 1)  # one_hot: batch * pids
        is_pos = self.repeat_instances(one_hot, dim=2).view(batch, -1)  # is_pos: batch * (pids * ins)
        is_neg = 1 - is_pos

        s_p = dist_mat * is_pos
        s_n = dist_mat * is_neg

        if self.type == 'cos':
            logit_p = -1.0 / self.temp * s_p + (-99999999.) * (1 - is_pos)
            logit_n = 1.0 / self.temp * (s_n + self.margin) + (-99999999.) * (1 - is_neg)
        elif self.type == 'circle':
            alpha_p = torch.clamp_min(-s_p.detach() + 1 + self.margin, min=0.)
            alpha_n = torch.clamp_min(s_n.detach() + self.margin, min=0.)
            delta_p = 1 - self.margin
            delta_n = self.margin

            logit_p = - 1.0 / self.temp * alpha_p * (s_p - delta_p) + (-99999999.) * (1 - is_pos)
            logit_n = 1.0 / self.temp * alpha_n * (s_n - delta_n) + (-99999999.) * (1 - is_neg)
        else:
            assert False, "invalid type {}".format(self.type)

        loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()
        return loss

