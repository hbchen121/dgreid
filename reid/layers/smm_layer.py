
import torch
import torch.nn as nn
import torch.nn.functional as F

from .adaNorm_layer import adaptive_instance_normalization as AdaIN
from .adaNorm_layer import multi_scale_adaIN as MSAdaIN
from .adaNorm_layer import AdaAttN, calc_mean_std


class StyleMixtureModule(nn.Module):
    """
    SMM: Style of domains Mixture Module
    """
    def __init__(self, dim, num_domains, half=True, with_in=False, lam=1.,
                 half_identity=False, num_instances=4, mix_layer='AdaIN',
                 multi_scale=None):
        super(StyleMixtureModule, self).__init__()
        self.num_domains = num_domains
        self.half = half
        self.with_in = with_in
        self.half_id = half_identity
        self.num_instances = num_instances
        assert -1.0 <= lam <= 1.0
        # -1 ~ 0: lam += 1, lam = random [lam, 1) [0,1)
        #  0 ~ 1: lam = lam
        self.lam = lam  # if lam is negative, using Random value
        self.instance_norm = nn.InstanceNorm2d(dim, affine=True) if with_in else nn.Identity()
        self.mix_layer = mix_layer
        if multi_scale and multi_scale[0] == 0:
            self.output_with_content = True
            multi_scale = multi_scale[1:]
        else:
            self.output_with_content = False
        self.multi_scale = multi_scale
        if mix_layer == 'AdaIN':
            self.style_mix_layer = MSAdaIN
            # self.style_mix_layer = AdaIN
        elif mix_layer == 'AdaAttN':
            self.style_mix_layer = AdaAttN(in_planes=dim)
        else:
            assert mix_layer in ['AdaIN', 'AdaAttN']

    def forward(self, x, identity=False):
        if not self.training or identity:
            return self.instance_norm(x)

        B, C, H, W = x.size()  # 48, 256, 64, 32
        assert (B % self.num_domains == 0), print(B, self.num_domains)
        with torch.no_grad():
            rand_indices = torch.randperm(B)
            x_shuffle = x[rand_indices]
        if self.multi_scale:
            x_domains = torch.chunk(x, self.num_domains, dim=0)
            x_domains_shuffle = torch.chunk(x_shuffle, self.num_domains, dim=0)
            x_output = []
            for x_domain, x_domain_shuffle in zip(x_domains, x_domains_shuffle):
                # 针对每一 domian 进行 style mix, 方便同一 domain 的数据 batch 连续
                x_domain_output = [x_domain] if self.output_with_content else []

                for scale in self.multi_scale:
                    # 针对每一 scale 分别 mix，然后串起来组成一个 domain 的 output
                    assert H % scale == 0
                    lam = 1 + self.lam * torch.rand(B // self.num_domains, 1, 1, 1).to(
                        x.device) if self.lam < 0 else self.lam
                    x_mix_scales = self.style_mix_layer(x_domain, x_domain_shuffle, scale, lam)
                    # x_domain_scale = torch.chunk(x_domain, scale, dim=2)  # in H
                    # x_domain_shuffle_scale = torch.chunk(x_domain_shuffle, scale, dim=2)
                    # lam = 1 + self.lam * torch.rand(B // self.num_domains, 1, 1, 1).to(x.device) if self.lam < 0 else self.lam
                    # x_mix_scales = []
                    # for i in range(scale):
                    #     # 针对 H 的每一 块进行 AdaIN
                    #     x_mix_scale, _, _ = self.style_mix_layer(x_domain_scale[i], x_domain_shuffle_scale[i], lam)
                    #     x_mix_scales.append(x_mix_scale)
                    # x_mix_scales = torch.cat(tuple(x_mix_scales), dim=2)
                    x_domain_output.append(x_mix_scales)
                x_domain_output = torch.cat(tuple(x_domain_output), dim=0)
                x_output.append(x_domain_output)
            x_output = torch.cat(tuple(x_output), dim=0)
            return x_output

        if not self.half:
            # uniform to [1 + self.lam, 1)
            # 1 + ((1 + self.lam) - 1) = 1 + self.lam
            lam = 1 + self.lam * torch.rand(B, 1, 1, 1).to(x.device) if self.lam < 0 else self.lam
            x_style_mix, mix_mean, mix_std = self.style_mix_layer(x, x_shuffle, lam)
            if self.mix_layer != 'AdaIN':
                mix_mean = torch.mean(mix_mean.view(B, C, -1), dim=2).view(B, C, 1, 1)
                mix_std = torch.mean(mix_std.view(B, C, -1), dim=2).view(B, C, 1, 1)
                style_mean, style_std = calc_mean_std(x_shuffle)
                return x_style_mix, mix_mean, mix_std, style_mean, style_std
            return x_style_mix
        x_domains = torch.chunk(x, self.num_domains, dim=0)
        x_domains_shuffle = torch.chunk(x_shuffle, self.num_domains, dim=0)
        x_domains_new = []
        x_mix_means = []
        x_mix_stds = []
        x_style = []
        for x_domain, x_domain_shuffle in zip(x_domains, x_domains_shuffle):
            if self.half_id:
                # device_num = torch.cuda.device_count()
                assert x_domain.size()[0] % self.num_instances == 0
                assert (x_domain.size()[0] // self.num_instances) % 2 == 0
                assert self.num_instances % 2 == 0
                x_domain_instance = x_domain.contiguous().view(-1, self.num_instances, C, H, W)
                x_domain_instance_shuffle = x_domain_shuffle.contiguous().view(-1, self.num_instances, C, H, W)
                x_domain_self, x_domain_mix = torch.chunk(x_domain_instance, 2, dim=1)
                _, x_domain_shuffle = torch.chunk(x_domain_instance_shuffle, 2, dim=1)
                x_domain_self = x_domain_self.contiguous().view(-1, C, H, W)
                x_domain_mix = x_domain_mix.contiguous().view(-1, C, H, W)
                x_domain_shuffle = x_domain_shuffle.contiguous().view(-1, C, H, W)
                x_domain_self = self.instance_norm(x_domain_self)
                B_ = x_domain_mix.size()[0]
                lam = 1 + self.lam * torch.rand(B_, 1, 1, 1).to(x.device) if self.lam < 0 else self.lam
                x_domain_mix, x_mix_mean, x_mix_std = self.style_mix_layer(x_domain_mix, x_domain_shuffle, lam)
                x_domain_self = x_domain_self.view(-1, self.num_instances // 2, C, H, W)
                x_domain_mix = x_domain_mix.view(-1, self.num_instances // 2, C, H, W)
                x_mix_means.append(x_mix_mean)
                x_mix_stds.append(x_mix_std)
                x_style.append(x_domain_shuffle)
                x_domain_new = torch.cat((x_domain_self, x_domain_mix), dim=1).view(-1, C, H, W)
                x_domains_new.append(x_domain_new)
            else:
                x_domain_self, x_domain_mix = torch.chunk(x_domain, 2, dim=0)
                _, x_domain_shuffle = torch.chunk(x_domain_shuffle, 2, dim=0)
                x_domain_self = self.instance_norm(x_domain_self)
                B_ = x_domain_mix.size()[0]
                lam = 1 + self.lam * torch.rand(B_, 1, 1, 1).to(x.device) if self.lam < 0 else self.lam
                x_domain_mix, x_mix_mean, x_mix_std = self.style_mix_layer(x_domain_mix, x_domain_shuffle, lam)
                x_mix_means.append(x_mix_mean)
                x_mix_stds.append(x_mix_std)
                x_style.append(x_domain_shuffle)
                x_domain_new = torch.cat((x_domain_self, x_domain_mix), dim=0)
                x_domains_new.append(x_domain_new)
        x_domains_new = torch.cat(tuple(x_domains_new), dim=0)
        if self.mix_layer != 'AdaIN':
            x_mix_means = torch.cat(tuple(x_mix_means), dim=0)
            x_mix_stds = torch.cat(tuple(x_mix_stds), dim=0)
            x_style = torch.cat(tuple(x_style), dim=0)
            mix_mean = torch.mean(x_mix_means.view(B // 2, C, -1), dim=2).view(B // 2, C, 1, 1)
            mix_std = torch.mean(x_mix_stds.view(B // 2, C, -1), dim=2).view(B // 2, C, 1, 1)
            style_mean, style_std = calc_mean_std(x_style)
            return x_domains_new, mix_mean, mix_std, style_mean, style_std
        else:
            del x_mix_means
            del x_mix_stds
            del x_style

        return x_domains_new


class MetaStyleMixtureModule(nn.Module):
    """
    SMM: Style of domains Mixture Module
    """
    def __init__(self, dim, num_domains, half=True, with_in=False, lam=1.,
                 half_identity=False, num_instances=4, mix_layer='AdaIN',
                 multi_scale=None, track_running_stats=False,
                 momentum=None):
        super(MetaStyleMixtureModule, self).__init__()
        self.num_domains = num_domains - 1
        self.half = half
        self.with_in = with_in
        self.half_id = half_identity
        self.num_instances = num_instances
        assert -1.0 <= lam <= 1.0
        # -1 ~ 0: lam += 1, lam = random [lam, 1) [0,1)
        #  0 ~ 1: lam = lam
        self.lam = lam  # if lam is negative, using Random value
        self.instance_norm = nn.InstanceNorm2d(dim, affine=True) if with_in else nn.Identity()
        self.mix_layer = mix_layer
        if multi_scale and multi_scale[0] == 0:
            self.output_with_content = True
            multi_scale = multi_scale[1:]
        else:
            self.output_with_content = False
        self.multi_scale = multi_scale
        if mix_layer == 'AdaIN':
            # self.style_mix_layer = AdaIN
            self.style_mix_layer = MSAdaIN
        elif mix_layer == 'AdaAttN':
            self.style_mix_layer = AdaAttN(in_planes=dim)
        else:
            assert mix_layer in ['AdaIN', 'AdaAttN']

        self.track_running_stats = track_running_stats
        # self.momentum = None
        self.momentum = momentum
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(dim))
            self.register_buffer('running_std', torch.ones(dim))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_std', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_running_stats()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_std.fill_(1)
            self.num_batches_tracked.zero_()

    def forward(self, x, meta_training=True):
        if not self.training or not meta_training:
            if self.training or meta_training:
                return self.instance_norm(x)
            if not self.training and self.track_running_stats:
                mean, std = calc_mean_std(x)
                size = x.size()
                normalized_feat = (x - mean.expand(size)) / std.expand(size)
                feat = normalized_feat * self.running_std.view(1, -1, 1, 1) + self.running_mean.view(1, -1, 1, 1)
                return feat
            return self.instance_norm(x)

        if self.training and self.track_running_stats:
            # use exponential moving average
            exponential_average_factor = self.momentum
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                # else:  # use exponential moving average
                #     exponential_average_factor = self.momentum
            with torch.no_grad():
                mean, std = calc_mean_std(x)
                mean = mean.mean([0, 2, 3])
                std = std.mean([0, 2, 3])
                self.running_mean = (exponential_average_factor * self.running_mean) + (
                            1.0 - exponential_average_factor) * mean  # .to(input.device)
                self.running_std = (exponential_average_factor * self.running_std) + (
                            1.0 - exponential_average_factor) * std

        B, C, H, W = x.size()  # 48, 256, 64, 32
        assert (B % self.num_domains == 0), print(B, self.num_domains)
        with torch.no_grad():
            rand_indices = torch.randperm(B)
            x_shuffle = x[rand_indices]
        if self.multi_scale:
            x_domains = torch.chunk(x, self.num_domains, dim=0)
            x_domains_shuffle = torch.chunk(x_shuffle, self.num_domains, dim=0)
            x_output = []
            for x_domain, x_domain_shuffle in zip(x_domains, x_domains_shuffle):
                # 针对每一 domian 进行 style mix, 方便同一 domain 的数据 batch 连续
                x_domain_output = [x_domain] if self.output_with_content else []
                for scale in self.multi_scale:
                    # 针对每一 scale 分别 mix，然后串起来组成一个 domain 的 output
                    lam = 1 + self.lam * torch.rand(B // self.num_domains, 1, 1, 1).to(
                        x.device) if self.lam < 0 else self.lam
                    x_mix_scales = self.style_mix_layer(x_domain, x_domain_shuffle, scale, lam)
                    # x_domain_scale = torch.chunk(x_domain, scale, dim=2)  # in H
                    # x_domain_shuffle_scale = torch.chunk(x_domain_shuffle, scale, dim=2)
                    # lam = 1 + self.lam * torch.rand(B // self.num_domains, 1, 1, 1).to(
                    #     x.device) if self.lam < 0 else self.lam
                    # x_mix_scales = []
                    # for i in range(scale):
                    #     # 针对 H 的每一 块进行 AdaIN
                    #     x_mix_scale, _, _ = self.style_mix_layer(x_domain_scale[i], x_domain_shuffle_scale[i], lam)
                    #     x_mix_scales.append(x_mix_scale)
                    # x_mix_scales = torch.cat(tuple(x_mix_scales), dim=2)
                    x_domain_output.append(x_mix_scales)
                x_domain_output = torch.cat(tuple(x_domain_output), dim=0)
                x_output.append(x_domain_output)
            x_output = torch.cat(tuple(x_output), dim=0)
            return x_output

        if not self.half:
            # uniform to [1 + self.lam, 1)
            # 1 + ((1 + self.lam) - 1) = 1 + self.lam
            lam = 1 + self.lam * torch.rand(B, 1, 1, 1).to(x.device) if self.lam < 0 else self.lam
            x_style_mix, mix_mean, mix_std = self.style_mix_layer(x, x_shuffle, lam)
            if self.mix_layer != 'AdaIN':
                mix_mean = torch.mean(mix_mean.view(B, C, -1), dim=2).view(B, C, 1, 1)
                mix_std = torch.mean(mix_std.view(B, C, -1), dim=2).view(B, C, 1, 1)
                style_mean, style_std = calc_mean_std(x_shuffle)
                return x_style_mix, mix_mean, mix_std, style_mean, style_std
            return x_style_mix
        x_domains = torch.chunk(x, self.num_domains, dim=0)
        x_domains_shuffle = torch.chunk(x_shuffle, self.num_domains, dim=0)
        x_domains_new = []
        x_mix_means = []
        x_mix_stds = []
        x_style = []
        for x_domain, x_domain_shuffle in zip(x_domains, x_domains_shuffle):
            if self.half_id:
                # device_num = torch.cuda.device_count()
                assert x_domain.size()[0] % self.num_instances == 0
                assert (x_domain.size()[0] // self.num_instances) % 2 == 0
                assert self.num_instances % 2 == 0
                x_domain_instance = x_domain.contiguous().view(-1, self.num_instances, C, H, W)
                x_domain_instance_shuffle = x_domain_shuffle.contiguous().view(-1, self.num_instances, C, H, W)
                x_domain_self, x_domain_mix = torch.chunk(x_domain_instance, 2, dim=1)
                _, x_domain_shuffle = torch.chunk(x_domain_instance_shuffle, 2, dim=1)
                x_domain_self = x_domain_self.contiguous().view(-1, C, H, W)
                x_domain_mix = x_domain_mix.contiguous().view(-1, C, H, W)
                x_domain_shuffle = x_domain_shuffle.contiguous().view(-1, C, H, W)
                x_domain_self = self.instance_norm(x_domain_self)
                B_ = x_domain_mix.size()[0]
                lam = 1 + self.lam * torch.rand(B_, 1, 1, 1).to(x.device) if self.lam < 0 else self.lam
                x_domain_mix, x_mix_mean, x_mix_std = self.style_mix_layer(x_domain_mix, x_domain_shuffle, lam)
                x_domain_self = x_domain_self.view(-1, self.num_instances // 2, C, H, W)
                x_domain_mix = x_domain_mix.view(-1, self.num_instances // 2, C, H, W)
                x_mix_means.append(x_mix_mean)
                x_mix_stds.append(x_mix_std)
                x_style.append(x_domain_shuffle)
                x_domain_new = torch.cat((x_domain_self, x_domain_mix), dim=1).view(-1, C, H, W)
                x_domains_new.append(x_domain_new)
            else:
                x_domain_self, x_domain_mix = torch.chunk(x_domain, 2, dim=0)
                _, x_domain_shuffle = torch.chunk(x_domain_shuffle, 2, dim=0)
                x_domain_self = self.instance_norm(x_domain_self)
                B_ = x_domain_mix.size()[0]
                lam = 1 + self.lam * torch.rand(B_, 1, 1, 1).to(x.device) if self.lam < 0 else self.lam
                x_domain_mix, x_mix_mean, x_mix_std = self.style_mix_layer(x_domain_mix, x_domain_shuffle, lam)
                x_mix_means.append(x_mix_mean)
                x_mix_stds.append(x_mix_std)
                x_style.append(x_domain_shuffle)
                x_domain_new = torch.cat((x_domain_self, x_domain_mix), dim=0)
                x_domains_new.append(x_domain_new)
        x_domains_new = torch.cat(tuple(x_domains_new), dim=0)
        if self.mix_layer != 'AdaIN':
            x_mix_means = torch.cat(tuple(x_mix_means), dim=0)
            x_mix_stds = torch.cat(tuple(x_mix_stds), dim=0)
            x_style = torch.cat(tuple(x_style), dim=0)
            mix_mean = torch.mean(x_mix_means.view(B // 2, C, -1), dim=2).view(B // 2, C, 1, 1)
            mix_std = torch.mean(x_mix_stds.view(B // 2, C, -1), dim=2).view(B // 2, C, 1, 1)
            style_mean, style_std = calc_mean_std(x_style)
            return x_domains_new, mix_mean, mix_std, style_mean, style_std
        else:
            del x_mix_means
            del x_mix_stds
            del x_style

        return x_domains_new


class StyleMixtureModule_backup(nn.Module):
    """
    SMM: Style of domains Mixture Module
    """
    def __init__(self, dim, num_domains, half=True, with_in=False, lam=1.):
        super(StyleMixtureModule_backup, self).__init__()
        self.num_domains = num_domains
        self.half = half
        self.with_in = with_in
        assert -1.0 <= lam <= 1.0
        # -1 ~ 0: lam += 1, lam = random [lam, 1) [0,1)
        #  0 ~ 1: lam = lam
        self.lam = lam  # if lam is negative, using Random value
        self.instance_norm = nn.InstanceNorm2d(dim, affine=True) if with_in else nn.Identity()

    def forward(self, x):
        if not self.training:
            return self.instance_norm(x)

        B, C, H, W = x.size()
        assert (B % self.num_domains == 0), print(B, self.num_domains)
        with torch.no_grad():
            rand_indices = torch.randperm(B)
            x_shuffle = x[rand_indices]
        if not self.half:
            # uniform to [1 + self.lam, 1)
            # 1 + ((1 + self.lam) - 1) = 1 + self.lam
            lam = 1 + self.lam * torch.rand(B, 1, 1, 1).to(x.device) if self.lam < 0 else self.lam
            x_style_mix = AdaIN(x, x_shuffle, lam)
            return x_style_mix
        x_domains = torch.chunk(x, self.num_domains, dim=0)
        x_domains_shuffle = torch.chunk(x_shuffle, self.num_domains, dim=0)
        x_domains_new = []
        for x_domain, x_domain_shuffle in zip(x_domains, x_domains_shuffle):
            x_domain_self, x_domain_mix = torch.chunk(x_domain, 2, dim=0)
            _, x_domain_shuffle = torch.chunk(x_domain_shuffle, 2, dim=0)
            x_domain_self = self.instance_norm(x_domain_self)
            B_ = x_domain_mix.size()[0]
            lam = 1 + self.lam * torch.rand(B_, 1, 1, 1).to(x.device) if self.lam < 0 else self.lam
            x_domain_mix = AdaIN(x_domain_mix, x_domain_shuffle, lam)
            x_domain_new = torch.cat((x_domain_self, x_domain_mix), dim=0)
            x_domains_new.append(x_domain_new)
        x_domains_new = torch.cat(tuple(x_domains_new), dim=0)
        return x_domains_new


class StyleMixtureModule_bakcup2(nn.Module):
    """
    SMM: Style of domains Mixture Module
    """
    def __init__(self, dim, num_domains, half=True, with_in=False, lam=1.,
                 half_identity=False, num_instances=4, mix_layer='AdaIN'):
        super(StyleMixtureModule, self).__init__()
        self.num_domains = num_domains
        self.half = half
        self.with_in = with_in
        self.half_id = half_identity
        self.num_instances = num_instances
        assert -1.0 <= lam <= 1.0
        # -1 ~ 0: lam += 1, lam = random [lam, 1) [0,1)
        #  0 ~ 1: lam = lam
        self.lam = lam  # if lam is negative, using Random value
        self.instance_norm = nn.InstanceNorm2d(dim, affine=True) if with_in else nn.Identity()
        self.mix_layer = mix_layer
        if mix_layer == 'AdaIN':
            self.style_mix_layer = AdaIN
        elif mix_layer == 'AdaAttN':
            self.style_mix_layer = AdaAttN(in_planes=dim)
        else:
            assert mix_layer in ['AdaIN', 'AdaAttN']

    def forward(self, x):
        if not self.training:
            return self.instance_norm(x)

        B, C, H, W = x.size()
        assert (B % self.num_domains == 0), print(B, self.num_domains)
        with torch.no_grad():
            rand_indices = torch.randperm(B)
            x_shuffle = x[rand_indices]
        if not self.half:
            # uniform to [1 + self.lam, 1)
            # 1 + ((1 + self.lam) - 1) = 1 + self.lam
            lam = 1 + self.lam * torch.rand(B, 1, 1, 1).to(x.device) if self.lam < 0 else self.lam
            x_style_mix = self.style_mix_layer(x, x_shuffle, lam)
            if self.mix_layer != 'AdaIN':
                return x_style_mix, x.detach(), x_style_mix, x_shuffle.detach()
            return x_style_mix
        x_domains = torch.chunk(x, self.num_domains, dim=0)
        x_domains_shuffle = torch.chunk(x_shuffle, self.num_domains, dim=0)
        x_domains_new = []
        x_content = []
        x_style_mix = []
        x_style = []
        for x_domain, x_domain_shuffle in zip(x_domains, x_domains_shuffle):
            if self.half_id:
                # device_num = torch.cuda.device_count()
                assert x_domain.size()[0] % self.num_instances == 0
                assert (x_domain.size()[0] // self.num_instances) % 2 == 0
                assert self.num_instances % 2 == 0
                x_domain_instance = x_domain.contiguous().view(-1, self.num_instances, C, H, W)
                x_domain_instance_shuffle = x_domain_shuffle.contiguous().view(-1, self.num_instances, C, H, W)
                x_domain_self, x_domain_mix = torch.chunk(x_domain_instance, 2, dim=1)
                _, x_domain_shuffle = torch.chunk(x_domain_instance_shuffle, 2, dim=1)
                x_domain_self = x_domain_self.contiguous().view(-1, C, H, W)
                x_domain_mix = x_domain_mix.contiguous().view(-1, C, H, W)
                x_domain_shuffle = x_domain_shuffle.contiguous().view(-1, C, H, W)
                x_domain_self = self.instance_norm(x_domain_self)
                B_ = x_domain_mix.size()[0]
                lam = 1 + self.lam * torch.rand(B_, 1, 1, 1).to(x.device) if self.lam < 0 else self.lam
                x_content.append(x_domain_mix)
                x_domain_mix = self.style_mix_layer(x_domain_mix, x_domain_shuffle, lam)
                x_domain_self = x_domain_self.view(-1, self.num_instances // 2, C, H, W)
                x_domain_mix = x_domain_mix.view(-1, self.num_instances // 2, C, H, W)
                x_style_mix.append(x_domain_mix)
                x_style.append(x_domain_shuffle)
                x_domain_new = torch.cat((x_domain_self, x_domain_mix), dim=1).view(-1, C, H, W)
                x_domains_new.append(x_domain_new)
            else:
                x_domain_self, x_domain_mix = torch.chunk(x_domain, 2, dim=0)
                _, x_domain_shuffle = torch.chunk(x_domain_shuffle, 2, dim=0)
                x_domain_self = self.instance_norm(x_domain_self)
                B_ = x_domain_mix.size()[0]
                lam = 1 + self.lam * torch.rand(B_, 1, 1, 1).to(x.device) if self.lam < 0 else self.lam
                x_content.append(x_domain_mix)
                x_domain_mix = self.style_mix_layer(x_domain_mix, x_domain_shuffle, lam)
                x_style_mix.append(x_domain_mix)
                x_style.append(x_domain_shuffle)
                x_domain_new = torch.cat((x_domain_self, x_domain_mix), dim=0)
                x_domains_new.append(x_domain_new)
        x_domains_new = torch.cat(tuple(x_domains_new), dim=0)
        if self.mix_layer != 'AdaIN':
            x_content = torch.cat(tuple(x_content), dim=0)
            x_style_mix = torch.cat(tuple(x_style_mix), dim=0)
            x_style = torch.cat(tuple(x_style), dim=0)
            return x_domains_new, x_content, x_style_mix, x_style
        else:
            del x_style_mix
            del x_style
        return x_domains_new


class StyleMixtureModule_backup3(nn.Module):
    """
    SMM: Style of domains Mixture Module
    """
    def __init__(self, dim, num_domains, half=True, with_in=False, lam=1.,
                 half_identity=False, num_instances=4, mix_layer='AdaIN'):
        super(StyleMixtureModule, self).__init__()
        self.num_domains = num_domains
        self.half = half
        self.with_in = with_in
        self.half_id = half_identity
        self.num_instances = num_instances
        assert -1.0 <= lam <= 1.0
        # -1 ~ 0: lam += 1, lam = random [lam, 1) [0,1)
        #  0 ~ 1: lam = lam
        self.lam = lam  # if lam is negative, using Random value
        self.instance_norm = nn.InstanceNorm2d(dim, affine=True) if with_in else nn.Identity()
        self.mix_layer = mix_layer
        if mix_layer == 'AdaIN':
            self.style_mix_layer = AdaIN
        elif mix_layer == 'AdaAttN':
            self.style_mix_layer = AdaAttN(in_planes=dim)
        else:
            assert mix_layer in ['AdaIN', 'AdaAttN']

    def forward(self, x):
        if not self.training:
            return self.instance_norm(x)

        B, C, H, W = x.size()
        assert (B % self.num_domains == 0), print(B, self.num_domains)
        with torch.no_grad():
            rand_indices = torch.randperm(B)
            x_shuffle = x[rand_indices]
        if not self.half:
            # uniform to [1 + self.lam, 1)
            # 1 + ((1 + self.lam) - 1) = 1 + self.lam
            lam = 1 + self.lam * torch.rand(B, 1, 1, 1).to(x.device) if self.lam < 0 else self.lam
            x_style_mix, mix_mean, mix_std = self.style_mix_layer(x, x_shuffle, lam)
            if self.mix_layer != 'AdaIN':
                mix_mean = torch.mean(mix_mean.view(B, C, -1), dim=2).view(B, C, 1, 1)
                mix_std = torch.mean(mix_std.view(B, C, -1), dim=2).view(B, C, 1, 1)
                style_mean, style_std = calc_mean_std(x_shuffle)
                return x_style_mix, mix_mean, mix_std, style_mean, style_std
            return x_style_mix
        x_domains = torch.chunk(x, self.num_domains, dim=0)
        x_domains_shuffle = torch.chunk(x_shuffle, self.num_domains, dim=0)
        x_domains_new = []
        x_mix_means = []
        x_mix_stds = []
        x_style = []
        for x_domain, x_domain_shuffle in zip(x_domains, x_domains_shuffle):
            if self.half_id:
                # device_num = torch.cuda.device_count()
                assert x_domain.size()[0] % self.num_instances == 0
                assert (x_domain.size()[0] // self.num_instances) % 2 == 0
                assert self.num_instances % 2 == 0
                x_domain_instance = x_domain.contiguous().view(-1, self.num_instances, C, H, W)
                x_domain_instance_shuffle = x_domain_shuffle.contiguous().view(-1, self.num_instances, C, H, W)
                x_domain_self, x_domain_mix = torch.chunk(x_domain_instance, 2, dim=1)
                _, x_domain_shuffle = torch.chunk(x_domain_instance_shuffle, 2, dim=1)
                x_domain_self = x_domain_self.contiguous().view(-1, C, H, W)
                x_domain_mix = x_domain_mix.contiguous().view(-1, C, H, W)
                x_domain_shuffle = x_domain_shuffle.contiguous().view(-1, C, H, W)
                x_domain_self = self.instance_norm(x_domain_self)
                B_ = x_domain_mix.size()[0]
                lam = 1 + self.lam * torch.rand(B_, 1, 1, 1).to(x.device) if self.lam < 0 else self.lam
                x_domain_mix, x_mix_mean, x_mix_std = self.style_mix_layer(x_domain_mix, x_domain_shuffle, lam)
                x_domain_self = x_domain_self.view(-1, self.num_instances // 2, C, H, W)
                x_domain_mix = x_domain_mix.view(-1, self.num_instances // 2, C, H, W)
                x_mix_means.append(x_mix_mean)
                x_mix_stds.append(x_mix_std)
                x_style.append(x_domain_shuffle)
                x_domain_new = torch.cat((x_domain_self, x_domain_mix), dim=1).view(-1, C, H, W)
                x_domains_new.append(x_domain_new)
            else:
                x_domain_self, x_domain_mix = torch.chunk(x_domain, 2, dim=0)
                _, x_domain_shuffle = torch.chunk(x_domain_shuffle, 2, dim=0)
                x_domain_self = self.instance_norm(x_domain_self)
                B_ = x_domain_mix.size()[0]
                lam = 1 + self.lam * torch.rand(B_, 1, 1, 1).to(x.device) if self.lam < 0 else self.lam
                x_domain_mix, x_mix_mean, x_mix_std = self.style_mix_layer(x_domain_mix, x_domain_shuffle, lam)
                x_mix_means.append(x_mix_mean)
                x_mix_stds.append(x_mix_std)
                x_style.append(x_domain_shuffle)
                x_domain_new = torch.cat((x_domain_self, x_domain_mix), dim=0)
                x_domains_new.append(x_domain_new)
        x_domains_new = torch.cat(tuple(x_domains_new), dim=0)
        if self.mix_layer != 'AdaIN':
            x_mix_means = torch.cat(tuple(x_mix_means), dim=0)
            x_mix_stds = torch.cat(tuple(x_mix_stds), dim=0)
            x_style = torch.cat(tuple(x_style), dim=0)
            mix_mean = torch.mean(x_mix_means.view(B // 2, C, -1), dim=2).view(B // 2, C, 1, 1)
            mix_std = torch.mean(x_mix_stds.view(B // 2, C, -1), dim=2).view(B // 2, C, 1, 1)
            style_mean, style_std = calc_mean_std(x_style)
            return x_domains_new, mix_mean, mix_std, style_mean, style_std
        else:
            del x_mix_means
            del x_mix_stds
            del x_style

        return x_domains_new


class StyleMixtureModule_backup(nn.Module):
    """
    SMM: Style of domains Mixture Module
    """
    def __init__(self, dim, num_domains, half=True, with_in=False, lam=1.,
                 half_identity=False, num_instances=4, mix_layer='AdaIN',
                 multi_scale=None):
        super(StyleMixtureModule_backup, self).__init__()
        self.num_domains = num_domains
        self.half = half
        self.with_in = with_in
        self.half_id = half_identity
        self.num_instances = num_instances
        assert -1.0 <= lam <= 1.0
        # -1 ~ 0: lam += 1, lam = random [lam, 1) [0,1)
        #  0 ~ 1: lam = lam
        self.lam = lam  # if lam is negative, using Random value
        self.instance_norm = nn.InstanceNorm2d(dim, affine=True) if with_in else nn.Identity()
        self.mix_layer = mix_layer
        if multi_scale and multi_scale[0] == 0:
            self.output_with_content = True
            multi_scale = multi_scale[1:]
        else:
            self.output_with_content = False
        self.multi_scale = multi_scale
        if mix_layer == 'AdaIN':
            self.style_mix_layer = AdaIN
        elif mix_layer == 'AdaAttN':
            self.style_mix_layer = AdaAttN(in_planes=dim)
        else:
            assert mix_layer in ['AdaIN', 'AdaAttN']

    def forward(self, x, identity=False):
        if not self.training or identity:
            return self.instance_norm(x)

        B, C, H, W = x.size()  # 48, 256, 64, 32
        assert (B % self.num_domains == 0), print(B, self.num_domains)
        with torch.no_grad():
            rand_indices = torch.randperm(B)
            x_shuffle = x[rand_indices]
        if self.multi_scale:
            x_domains = torch.chunk(x, self.num_domains, dim=0)
            x_domains_shuffle = torch.chunk(x_shuffle, self.num_domains, dim=0)
            x_output = []
            for x_domain, x_domain_shuffle in zip(x_domains, x_domains_shuffle):
                # 针对每一 domian 进行 style mix, 方便同一 domain 的数据 batch 连续
                x_domain_output = [x_domain] if self.output_with_content else []
                for scale in self.multi_scale:
                    # 针对每一 scale 分别 mix，然后串起来组成一个 domain 的 output
                    assert H % scale == 0
                    x_domain_scale = torch.chunk(x_domain, scale, dim=2)  # in H
                    x_domain_shuffle_scale = torch.chunk(x_domain_shuffle, scale, dim=2)
                    lam = 1 + self.lam * torch.rand(B // self.num_domains, 1, 1, 1).to(x.device) if self.lam < 0 else self.lam
                    x_mix_scales = []
                    for i in range(scale):
                        # 针对 H 的每一 块进行 AdaIN
                        x_mix_scale, _, _ = self.style_mix_layer(x_domain_scale[i], x_domain_shuffle_scale[i], lam)
                        x_mix_scales.append(x_mix_scale)
                    x_mix_scales = torch.cat(tuple(x_mix_scales), dim=2)
                    x_domain_output.append(x_mix_scales)
                x_domain_output = torch.cat(tuple(x_domain_output), dim=0)
                x_output.append(x_domain_output)
            x_output = torch.cat(tuple(x_output), dim=0)
            return x_output

        if not self.half:
            # uniform to [1 + self.lam, 1)
            # 1 + ((1 + self.lam) - 1) = 1 + self.lam
            lam = 1 + self.lam * torch.rand(B, 1, 1, 1).to(x.device) if self.lam < 0 else self.lam
            x_style_mix, mix_mean, mix_std = self.style_mix_layer(x, x_shuffle, lam)
            if self.mix_layer != 'AdaIN':
                mix_mean = torch.mean(mix_mean.view(B, C, -1), dim=2).view(B, C, 1, 1)
                mix_std = torch.mean(mix_std.view(B, C, -1), dim=2).view(B, C, 1, 1)
                style_mean, style_std = calc_mean_std(x_shuffle)
                return x_style_mix, mix_mean, mix_std, style_mean, style_std
            return x_style_mix
        x_domains = torch.chunk(x, self.num_domains, dim=0)
        x_domains_shuffle = torch.chunk(x_shuffle, self.num_domains, dim=0)
        x_domains_new = []
        x_mix_means = []
        x_mix_stds = []
        x_style = []
        for x_domain, x_domain_shuffle in zip(x_domains, x_domains_shuffle):
            if self.half_id:
                # device_num = torch.cuda.device_count()
                assert x_domain.size()[0] % self.num_instances == 0
                assert (x_domain.size()[0] // self.num_instances) % 2 == 0
                assert self.num_instances % 2 == 0
                x_domain_instance = x_domain.contiguous().view(-1, self.num_instances, C, H, W)
                x_domain_instance_shuffle = x_domain_shuffle.contiguous().view(-1, self.num_instances, C, H, W)
                x_domain_self, x_domain_mix = torch.chunk(x_domain_instance, 2, dim=1)
                _, x_domain_shuffle = torch.chunk(x_domain_instance_shuffle, 2, dim=1)
                x_domain_self = x_domain_self.contiguous().view(-1, C, H, W)
                x_domain_mix = x_domain_mix.contiguous().view(-1, C, H, W)
                x_domain_shuffle = x_domain_shuffle.contiguous().view(-1, C, H, W)
                x_domain_self = self.instance_norm(x_domain_self)
                B_ = x_domain_mix.size()[0]
                lam = 1 + self.lam * torch.rand(B_, 1, 1, 1).to(x.device) if self.lam < 0 else self.lam
                x_domain_mix, x_mix_mean, x_mix_std = self.style_mix_layer(x_domain_mix, x_domain_shuffle, lam)
                x_domain_self = x_domain_self.view(-1, self.num_instances // 2, C, H, W)
                x_domain_mix = x_domain_mix.view(-1, self.num_instances // 2, C, H, W)
                x_mix_means.append(x_mix_mean)
                x_mix_stds.append(x_mix_std)
                x_style.append(x_domain_shuffle)
                x_domain_new = torch.cat((x_domain_self, x_domain_mix), dim=1).view(-1, C, H, W)
                x_domains_new.append(x_domain_new)
            else:
                x_domain_self, x_domain_mix = torch.chunk(x_domain, 2, dim=0)
                _, x_domain_shuffle = torch.chunk(x_domain_shuffle, 2, dim=0)
                x_domain_self = self.instance_norm(x_domain_self)
                B_ = x_domain_mix.size()[0]
                lam = 1 + self.lam * torch.rand(B_, 1, 1, 1).to(x.device) if self.lam < 0 else self.lam
                x_domain_mix, x_mix_mean, x_mix_std = self.style_mix_layer(x_domain_mix, x_domain_shuffle, lam)
                x_mix_means.append(x_mix_mean)
                x_mix_stds.append(x_mix_std)
                x_style.append(x_domain_shuffle)
                x_domain_new = torch.cat((x_domain_self, x_domain_mix), dim=0)
                x_domains_new.append(x_domain_new)
        x_domains_new = torch.cat(tuple(x_domains_new), dim=0)
        if self.mix_layer != 'AdaIN':
            x_mix_means = torch.cat(tuple(x_mix_means), dim=0)
            x_mix_stds = torch.cat(tuple(x_mix_stds), dim=0)
            x_style = torch.cat(tuple(x_style), dim=0)
            mix_mean = torch.mean(x_mix_means.view(B // 2, C, -1), dim=2).view(B // 2, C, 1, 1)
            mix_std = torch.mean(x_mix_stds.view(B // 2, C, -1), dim=2).view(B // 2, C, 1, 1)
            style_mean, style_std = calc_mean_std(x_style)
            return x_domains_new, mix_mean, mix_std, style_mean, style_std
        else:
            del x_mix_means
            del x_mix_stds
            del x_style

        return x_domains_new

