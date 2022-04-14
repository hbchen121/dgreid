import torch
import torch.nn.functional as F
from torch.nn import init
from torch import nn, autograd
import numpy as np
from ..loss import adv_loss
from .memory import MemoryClassifier, mc
from .softmaxs import cosSoftmax
from ..loss.mmd import mmd as mmd_loss
from ..loss.k_moment import moment_distance
import math


def non_overlap_choice(arr, candidate_arr, size, device):
    # 从 candidate_arr 随机取 size 个，不能和arr重复，且自身不重复
    arr = arr.cpu().numpy()
    for value in arr:
        candidate_arr = candidate_arr[candidate_arr != value]
    choice = np.random.choice(candidate_arr, size=size, replace=False)
    choice = torch.from_numpy(choice).to(device)
    return choice


class PartialMemory(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, features, mask, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes, mask)
        mask_features = ctx.features * mask
        outputs = inputs.mm(mask_features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes, mask = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            # ctx.features: num_pid * dim
            # mask: num_pid * 1 (one_hot)
            # mask_features: num_pid(部分为0) * dim
            mask_features = ctx.features * mask
            grad_inputs = grad_outputs.mm(mask_features)

        return grad_inputs, None, None, None, None


def pm(inputs, indexes, features, mask, momentum=0.5):
    return PartialMemory.apply(inputs, indexes, features, mask, torch.Tensor([momentum]).to(inputs.device))


def MaskCrossEntropy(logits, targets, mask, eps=1e-5):
    x = logits
    x = mask * x
    maxes = torch.max(x, 1, keepdim=True)[0]
    x_exp = torch.exp(x - maxes)
    x_exp = mask * x_exp
    # 正常情况下所有的 0 被 exp(0) 为 1， 参与 LSE 运算
    # 将 logsumexp 中 非mask 的部分乘为 0，则不参与 sum，因此只对 mask 部分进行 soft的sum底部的计算
    x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
    outputs = x_exp / x_exp_sum
    log_probs = torch.log(outputs + eps)
    log_probs = mask * log_probs
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    loss = (- targets * log_probs).mean(0).sum()
    return loss


class MixtureDomainMemory(nn.Module):
    def __init__(self, num_features, num_domains, domain_pids=None, temp=0.05, margin=0.,
                 momentum=0.2, num_instances=4, mix_domain_size=600, equality_mixture=True,
                 dynamic_momentum=0, decay=0.9999):
        super(MixtureDomainMemory, self).__init__()
        self.num_features = num_features
        assert len(domain_pids) == num_domains
        self.num_domains = num_domains
        self.domain_pids = domain_pids
        self.pidSum = [0]
        sum = 0
        for num in domain_pids:
            sum += num
            self.pidSum.append(sum)
        num_pids = self.pidSum[-1]
        self.num_pids = num_pids
        # self.relabel = ReLabel(self.domain_pids)
        self.momentum = momentum
        self.temp = temp
        self.margin = margin
        self.num_instances = num_instances
        self.mix_domain_size = mix_domain_size

        if dynamic_momentum > 0:
            self.decay = lambda x, m: decay * (1 - math.exp(-x / dynamic_momentum))
        else:
            self.decay = lambda x, m: m

        self.register_buffer('features', torch.zeros(num_pids, num_features))
        self.register_buffer('labels', torch.zeros(num_pids).long())

    def init(self, features, labels):
        """
        f: pid * dim, labels: pid * 1
        """
        assert features.shape == self.features.shape, print(features.shape, self.features.shape)
        self.features = features
        self.labels = labels

    def MomentumUpdate(self, inputs, indexes, mean_update=False):
        """
        inputs: batch(per_batch * num_domains) * dim, indexes: (per_batch * num_domains) * 1
        """
        # momentum update
        for x, y in zip(inputs, indexes):
            self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x
            self.features[y] = self.features[y] / self.features[y].norm()
    
    def domain_mask_back(self, domain_idx, targets):
        if domain_idx >= 0 and domain_idx < self.num_domains:
            len = self.domain_pids[domain_idx]
            start = self.pidSum[domain_idx]
            idxs = torch.arange(start, start + len)
        else:
            # import IPython
            # IPython.embed()
            batch_size = targets.shape[0]
            batch_idx = torch.arange(0, batch_size, self.num_instances)
            batch_idx = targets[batch_idx]
            if self.equality_mixture:
                per_domain_size = self.mix_domain_size // self.num_domains
                per_domain_choice = []
                for j, num_pid in enumerate(self.domain_pids):
                    start = self.pidSum[j]
                    domain_choice = np.random.choice(np.arange(start, start + num_pid), size=per_domain_size)
                    per_domain_choice.append(torch.from_numpy(domain_choice))
                idxs = torch.cat(tuple(per_domain_choice), dim=0)
            else:
                domain_choice = np.random.choice(np.arange(self.num_pids), size=self.mix_domain_size)
                domain_choice = torch.from_numpy(domain_choice)
                idxs = torch.cat((batch_idx, domain_choice), dim=0)
        num_pid, dim = self.features.shape
        mask = F.one_hot(idxs, num_pid).to(self.features.device)
        mask = mask.sum(dim=0).view(1, -1)  # 1 * num_pid
        mask = torch.clamp_max(mask, 1)
        return mask

    def forward_back(self, inputs, targets, domain_idx):
        # B, C = inputs.shape  # C -> num_pid
        logits = mc(inputs, targets, self.features, self.momentum)  # B * num_pids
        domain_mask = self.domain_mask(domain_idx, targets)  # 1 * num_pids
        loss = MaskCrossEntropy(logits, targets, 1.0 / self.temp, self.margin, domain_mask)
        # loss = F.cross_entropy(logits, targets)
        return loss

    def domain_mask(self, domain_idx, targets, logits):
        batch_size = targets.shape[0]
        if domain_idx >= 0 and domain_idx < self.num_domains:
            # 为各个 domain 选择 idx (standard mde memory loss calculate)
            len = self.domain_pids[domain_idx]
            start = self.pidSum[domain_idx]
            idxs = torch.arange(start, start + len)
        elif domain_idx == -1:
            # 将 所有domain 视为一个 domain
            idxs = torch.arange(0, self.num_pids)
        elif domain_idx == -2 or domain_idx == -3:
            # 平均随机 与 全部随机
            batch_idx = torch.arange(0, batch_size, self.num_instances)
            batch_idx = targets[batch_idx]
            batch_idx = torch.unique(batch_idx)
            if domain_idx == -2:
                # -2: domain平等，从各个 domain 中各自随机取出 size/ num_domains 个 pid
                per_domain_size = self.mix_domain_size // self.num_domains
                per_domain_choice = [batch_idx]
                for j, num_pid in enumerate(self.domain_pids):
                    start = self.pidSum[j]
                    domain_choice = non_overlap_choice(batch_idx, np.arange(start, start + num_pid),
                                                       per_domain_size, targets.device)
                    per_domain_choice.append(domain_choice)
                idxs = torch.cat(tuple(per_domain_choice), dim=0)
            else:
                # -3: 从所有domain 组成的大 domain 中随机取出 size 个 pid
                domain_choice = non_overlap_choice(batch_idx, np.arange(self.num_pids),
                                                   self.mix_domain_size, targets.device)
                idxs = torch.cat((batch_idx, domain_choice), dim=0)
        elif domain_idx == -4 or domain_idx == -5:
            logits = logits.sum(dim=0)  # pid * 1
            batch_idx = torch.arange(0, batch_size, self.num_instances)
            batch_idx = targets[batch_idx]
            batch_idx = torch.unique(batch_idx)
            if domain_idx == -4:
                per_domain_size = self.mix_domain_size // self.num_domains
                per_domain_choice = [batch_idx]
                for j, num_pid in enumerate(self.domain_pids):
                    start = self.pidSum[j]
                    _, domain_choice = torch.topk(logits[start: start + num_pid], per_domain_size)
                    domain_choice += start
                    per_domain_choice.append(domain_choice)
                idxs = torch.cat(tuple(per_domain_choice), dim=0)
            else:
                _, domain_choice = torch.topk(logits, self.mix_domain_size)
                idxs = torch.cat((batch_idx, domain_choice), dim=0)
            idxs = torch.unique(idxs)
        elif domain_idx == -6 or domain_idx == -7:
            # 平均随机 与 全部随机，数量为 总pid 的分数
            batch_idx = torch.arange(0, batch_size, self.num_instances)
            batch_idx = targets[batch_idx]
            batch_idx = torch.unique(batch_idx)
            if domain_idx == -6:
                # -2: domain平等，从各个 domain 中各自随机取出 1/n 的 id
                per_domain_choice = [batch_idx]
                for j, num_pid in enumerate(self.domain_pids):
                    start = self.pidSum[j]
                    per_domain_size = int(num_pid * self.mix_domain_size)
                    assert per_domain_size > 0
                    domain_choice = non_overlap_choice(batch_idx, np.arange(start, start + num_pid),
                                                       per_domain_size, targets.device)
                    per_domain_choice.append(domain_choice)
                idxs = torch.cat(tuple(per_domain_choice), dim=0)
            else:
                # -3: 从所有domain 组成的大 domain 中随机取出 1/n 个 pid
                domain_size = int(self.num_pids * self.mix_domain_size)
                assert domain_size > 0
                domain_choice = non_overlap_choice(batch_idx, np.arange(self.num_pids),
                                                   domain_size, targets.device)
                idxs = torch.cat((batch_idx, domain_choice), dim=0)
        idxs, _ = idxs.sort()

        # convert label to new label
        one_hot = torch.zeros(batch_size, self.num_pids, device=targets.device, dtype=targets.dtype)
        one_hot.scatter_(1, targets.unsqueeze(1), 1)
        one_hot = one_hot[:, idxs]
        targets_new = (one_hot == 1).nonzero(as_tuple=False)
        targets_new = targets_new[:, 1]
        assert targets_new.shape == targets.shape

        # mask
        num_pid, dim = self.features.shape
        mask = F.one_hot(idxs, num_pid).to(self.features.device)
        mask = mask.sum(dim=0).view(1, -1)  # 1 * num_pid
        return mask, idxs, targets_new

    def forward(self, inputs, targets, domain_idx, iter=0):
        # B, C = inputs.shape  # C -> num_pid
        self.momentum = self.decay(iter, self.momentum)

        # logits = pm(inputs, targets, self.features, domain_mask, self.momentum)  # B * num_pids
        logits = mc(inputs, targets, self.features, self.momentum)  # B * num_pids

        domain_mask, idxs, targets_new = self.domain_mask(domain_idx, targets, logits.detach())  # 1 * num_pids

        if self.margin > 0:
            logits = cosSoftmax(logits, targets, self.margin)
        logits = logits / self.temp

        logits = logits[:, idxs]
        loss = F.cross_entropy(logits, targets_new)

        # loss = MaskCrossEntropy(logits, targets, domain_mask)
        return loss

    def mmd(self, inputs, targets):
        batch_size = inputs.shape[0]
        # 相同 id 取 mean
        n_pid = batch_size // self.num_instances
        inputs = torch.stack(torch.chunk(inputs, n_pid, dim=0), dim=0)  # pid * intances * dim
        inputs = inputs.mean(dim=1)  # pid_batch * dim
        pids_mean_idx = torch.arange(0, batch_size, self.num_instances)
        targets = targets[pids_mean_idx]  # pid_batch * 1

        # pid_batch * dim -> pid_all * dim
        inputs = inputs.unsqueeze(dim=1).repeat(1, self.num_pids, 1)  # pid_bat * pid_all * dim
        mask_targets = F.one_hot(targets, self.num_pids).to(targets.device)  # pid_batch * pid_all
        mask_targets = mask_targets.unsqueeze(dim=-1)  # pid_batch * pid_all * 1
        mask_inputs = inputs * mask_targets  # pid_bat * pid_all * dim, 每一pid_bat中对于的 pid_all 中只保留一个
        mask_inputs = mask_inputs.sum(dim=0)  # pid_all * dim

        features = self.features.clone()  # pid_all * dim
        features[targets] = 0

        features = features + mask_inputs  # 置换 target 的 feat
        domain_features = []
        for j, num_pid in enumerate(self.domain_pids):
            start = self.pidSum[j]
            domain_features.append(features[start: start + num_pid])

        loss = 0.
        for i in range(self.num_domains):
            for j in range(i, self.num_domains):
                loss = loss + mmd_loss(domain_features[i], domain_features[j])

        return loss

    def moment_loss(self, inputs, targets, k, size):
        batch_size = inputs.shape[0]
        if size == 0:
            # 仅在本批次内进行矩损失
            pass
        else:
            # 在 memory 范围内进行矩损失计算
            pass
            # 先把 batch 内平均
            # 相同 id 取 mean
            n_pid = batch_size // self.num_instances
            inputs = torch.stack(torch.chunk(inputs, n_pid, dim=0), dim=0)  # pid * intances * dim
            inputs = inputs.mean(dim=1)  # pid_batch * dim
            pids_mean_idx = torch.arange(0, batch_size, self.num_instances)
            targets = targets[pids_mean_idx]  # pid_batch * 1

            # pid_batch * dim -> pid_all * dim
            inputs = inputs.unsqueeze(dim=1).repeat(1, self.num_pids, 1)  # pid_bat * pid_all * dim
            mask_targets = F.one_hot(targets, self.num_pids).to(targets.device)  # pid_batch * pid_all
            mask_targets = mask_targets.unsqueeze(dim=-1)  # pid_batch * pid_all * 1
            mask_inputs = inputs * mask_targets  # pid_bat * pid_all * dim, 每一pid_bat中对于的 pid_all 中只保留一个
            mask_inputs = mask_inputs.sum(dim=0)  # pid_all * dim

            features = self.features.clone()  # pid_all * dim
            features[targets] = 0

            features = features + mask_inputs  # 置换 target 的 feat
            domain_features = []
            for j, num_pid in enumerate(self.domain_pids):
                start = self.pidSum[j]
                domain_features.append(features[start: start + num_pid])

        loss = 0.
        for i in range(self.num_domains):
            for j in range(i, self.num_domains):
                loss = loss + moment_distance(domain_features[i], domain_features[j], k)

        return loss