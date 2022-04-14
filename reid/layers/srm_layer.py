# encoding: utf-8
import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter


class SRMLayer(nn.Module):
    """
    SRM : A Style-based Recalibration Module for Convolutional Neural Networks
    form: https://github.com/hyunjaelee410/style-based-recalibration-module.git
    """

    def __init__(self, channel):
        super(SRMLayer, self).__init__()

        self.cfc = Parameter(torch.Tensor(channel, 2))
        self.cfc.data.fill_(0)

        self.bn = nn.BatchNorm2d(channel)
        self.activation = nn.Sigmoid()

        setattr(self.cfc, 'srm_param', True)
        setattr(self.bn.weight, 'srm_param', True)
        setattr(self.bn.bias, 'srm_param', True)

    def _style_pooling(self, x, eps=1e-5):
        N, C, _, _ = x.size()

        channel_mean = x.view(N, C, -1).mean(dim=2, keepdim=True)
        channel_var = x.view(N, C, -1).var(dim=2, keepdim=True) + eps
        channel_std = channel_var.sqrt()

        t = torch.cat((channel_mean, channel_std), dim=2)
        return t

    def _style_integration(self, t):
        z = t * self.cfc[None, :, :]  # B x C x 2
        z = torch.sum(z, dim=2)[:, :, None, None]  # B x C x 1 x 1

        z_hat = self.bn(z)
        g = self.activation(z_hat)

        return g

    def forward(self, x):
        # B x C x 2
        t = self._style_pooling(x)

        # B x C x 1 x 1
        g = self._style_integration(t)

        return x * g


class CSRMLayer(nn.Module):
    """
    CSRM : A Cross Style-based Recalibration Module
    """

    def __init__(self, channel, eps=1e-5, momentum=0.1, track_running_stats=True):
        super(CSRMLayer, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        # cross-style pooling parameters
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(channel))
            self.register_buffer('running_var', torch.ones(channel))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_running_stats()

        self.cfc = Parameter(torch.Tensor(channel, 4))
        self.cfc.data.fill_(0)

        self.bn = nn.BatchNorm2d(channel)
        self.activation = nn.Sigmoid()

        setattr(self.cfc, 'srm_param', True)
        setattr(self.bn.weight, 'srm_param', True)
        setattr(self.bn.bias, 'srm_param', True)

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def _batch_style_pooling(self, x):
        # batch style
        if self.training:
            batch_mean = x.mean(dim=[0, 2, 3])
            batch_var = x.var(dim=[0, 2, 3])
            if self.track_running_stats:
                self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * batch_var
        else:
            if self.track_running_stats and self.num_batches_tracked.item() != 0:
                batch_mean = self.running_mean
                batch_var = self.running_var
            else:
                batch_mean = x.mean(dim=[0, 2, 3])
                batch_var = x.var(dim=[0, 2, 3])
        batch_std = (batch_var + self.eps).sqrt()
        return batch_mean, batch_std

    def _style_pooling(self, x):
        N, C, _, _ = x.size()
        # instance style
        channel_mean = x.view(N, C, -1).mean(dim=2, keepdim=True)
        channel_var = x.view(N, C, -1).var(dim=2, keepdim=True) + self.eps
        channel_std = channel_var.sqrt()

        batch_mean, batch_std = self._batch_style_pooling(x)  # C * 1
        batch_mean = batch_mean.view([1, C, 1]).expand_as(channel_mean)
        batch_std = batch_std.view([1, C, 1]).expand_as(channel_mean)

        t = torch.cat((channel_mean, channel_std, batch_mean, batch_std), dim=2)
        return t

    def _style_integration(self, t):
        z = t * self.cfc[None, :, :]  # B x C x 4
        z = torch.sum(z, dim=2)[:, :, None, None]  # B x C x 1 x 1

        z_hat = self.bn(z)
        g = self.activation(z_hat)

        return g

    def forward(self, x):
        # B x C x 4
        t = self._style_pooling(x)

        # B x C x 1 x 1
        g = self._style_integration(t)

        return x * g
