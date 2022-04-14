
import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_mean_std(feat, eps=1e-5):
    # from: https://github.com/naoto0804/pytorch-AdaIN/blob/3a5946bbab89eddf77190e996a7ab1a6ade94584/function.py#L15
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat, lam=1.):
    # AdaIN
    # modified from: https://github.com/naoto0804/pytorch-AdaIN/blob/3a5946bbab89eddf77190e996a7ab1a6ade94584/function.py#L15
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)

    mix_style_std = lam * style_std.expand(size) + (1 - lam) * content_std.expand(size)
    mix_style_mean = lam * style_mean.expand(size) + (1 - lam) * content_mean.expand(size)

    return normalized_feat * mix_style_std + mix_style_mean, mix_style_mean, mix_style_std


def multi_scale_adaIN(content_feat, style_feat, scale=1, lam=1.):

    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()

    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)

    ms_style_feat = torch.chunk(style_feat, scale, dim=2)
    size = ms_style_feat[0].size()
    # ms_style = [(calc_mean_std(i_style_feat)) for i_style_feat in ms_style_feat]
    mix_style_std = []
    mix_style_mean = []
    for i_style_feat in ms_style_feat:
        i_style_mean, i_style_std = calc_mean_std(i_style_feat)
        i_mix_style_std = lam * i_style_std.expand(size) + (1 - lam) * content_std.expand(size)
        i_mix_style_mean = lam * i_style_mean.expand(size) + (1 - lam) * content_mean.expand(size)
        mix_style_std.append(i_mix_style_std)
        mix_style_mean.append(i_mix_style_mean)
    mix_style_std = torch.cat(mix_style_std, dim=2)
    mix_style_mean = torch.cat(mix_style_mean, dim=2)

    return normalized_feat * mix_style_std + mix_style_mean


def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


class AdaAttN(nn.Module):

    def __init__(self, in_planes, key_planes=None):
        super(AdaAttN, self).__init__()
        if key_planes is None:
            key_planes = in_planes
        self.f = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.g = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)

    def forward(self, content, style, lam=0.):
        # cont, style, key with the same size
        content_key, style_key = content.detach(), style.detach()
        content_key, style_key = mean_variance_norm(content_key), mean_variance_norm(style_key)
        F = self.f(content_key)
        G = self.g(style_key)
        H = self.h(style)
        b, _, h, w = G.size()
        G = G.view(b, -1, w * h).contiguous()
        style_flat = H.view(b, -1, w * h).transpose(1, 2).contiguous()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        S = torch.bmm(F, G)
        # S: b, n_c, n_s
        S = self.sm(S)
        # mean: b, n_c, c
        mean = torch.bmm(S, style_flat)
        # std: b, n_c, c
        std = torch.sqrt(torch.relu(torch.bmm(S, style_flat ** 2) - mean ** 2))
        # mean, std: b, c, h, w
        mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return std * mean_variance_norm(content) + mean, mean, std
        # F = self.f(content_key)
        # G = self.g(style_key)
        # H = self.h(style)
        # b, _, h_g, w_g = G.size()
        # G = G.view(b, -1, w_g * h_g).contiguous()
        # if w_g * h_g > self.max_sample:
        #     if seed is not None:
        #         torch.manual_seed(seed)
        #     index = torch.randperm(w_g * h_g).to(content.device)[:self.max_sample]
        #     G = G[:, :, index]
        #     style_flat = H.view(b, -1, w_g * h_g)[:, :, index].transpose(1, 2).contiguous()
        # else:
        #     style_flat = H.view(b, -1, w_g * h_g).transpose(1, 2).contiguous()
        # b, _, h, w = F.size()
        # F = F.view(b, -1, w * h).permute(0, 2, 1)
        # S = torch.bmm(F, G)
        # # S: b, n_c, n_s
        # S = self.sm(S)
        # # mean: b, n_c, c
        # mean = torch.bmm(S, style_flat)
        # # std: b, n_c, c
        # std = torch.sqrt(torch.relu(torch.bmm(S, style_flat ** 2) - mean ** 2))
        # # mean, std: b, c, h, w
        # mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        # std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        # return std * mean_variance_norm(content) + mean