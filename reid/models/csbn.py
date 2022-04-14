import torch
import torch.nn as nn


# 使用 Target domain 的 \beta, \gamma 进行
class CrossBN2d(nn.Module):
    def __init__(self, planes, affine=False):
        super(CrossBN2d, self).__init__()
        self.BN = nn.BatchNorm2d(planes, affine=affine)

    def forward(self, x, weight, bias):
        weight = nn.Parameter(weight.data.clone(), requires_grad=False)
        bias = nn.Parameter(bias.data.clone(), requires_grad=False)
        self.BN.weight = weight
        self.BN.bias = bias
        return self.BN(x)


class CrossBN1d(nn.Module):
    def __init__(self, planes, affine=False):
        super(CrossBN1d, self).__init__()
        self.BN = nn.BatchNorm1d(planes, affine=affine)

    def forward(self, x, weight, bias):
        weight = nn.Parameter(weight.data.clone(), requires_grad=False)
        bias = nn.Parameter(bias.data.clone(), requires_grad=False)
        self.BN.weight = weight
        self.BN.bias = bias
        return self.BN(x)


# Cross-Domain BatchNorm
class CDBN2d(nn.Module):
    def __init__(self, planes):
        super(CDBN2d, self).__init__()
        self.num_features = planes
        self.BN_S = CrossBN2d(planes)
        self.BN_T = nn.BatchNorm2d(planes)

    def forward(self, x):
        if (not self.training):
            return self.BN_T(x)

        bs = x.size(0)
        assert (bs%2==0)
        split = torch.split(x, int(bs/2), 0)
        out2 = self.BN_T(split[1].contiguous())
        # import IPython
        # IPython.embed()
        out1 = self.BN_S(split[0].contiguous(), self.BN_T.weight, self.BN_T.bias)
        out = torch.cat((out1, out2), 0)
        return out


class CDBN1d(nn.Module):
    def __init__(self, planes):
        super(CDBN1d, self).__init__()
        self.num_features = planes
        self.BN_S = CrossBN1d(planes)
        self.BN_T = nn.BatchNorm1d(planes)

    def forward(self, x):
        if (not self.training):
            return self.BN_T(x)

        bs = x.size(0)
        assert (bs%2==0)
        split = torch.split(x, int(bs/2), 0)
        out2 = self.BN_T(split[1].contiguous())
        out1 = self.BN_S(split[0].contiguous(), self.BN_T.weight, self.BN_T.bias)
        out = torch.cat((out1, out2), 0)
        return out



def convert_csbn(model):
    for _, (child_name, child) in enumerate(model.named_children()):
        assert(not next(model.parameters()).is_cuda)
        if isinstance(child, nn.BatchNorm2d):
            m = CDBN2d(child.num_features)
            state = child.state_dict()
            state.pop('weight')
            state.pop('bias')
            m.BN_S.BN.load_state_dict(state)
            # m.BN_S.load_state_dict(child.state_dict())
            m.BN_T.load_state_dict(child.state_dict())
            setattr(model, child_name, m)
        elif isinstance(child, nn.BatchNorm1d) and child_name!='d_bn1':
            m = CDBN1d(child.num_features)
            state = child.state_dict()
            state.pop('weight')
            state.pop('bias')
            m.BN_S.BN.load_state_dict(state)
            # m.BN_S.load_state_dict(child.state_dict())
            m.BN_T.load_state_dict(child.state_dict())
            setattr(model, child_name, m)
        else:
            convert_csbn(child)

def convert_bn(model, use_target=True):
    for _, (child_name, child) in enumerate(model.named_children()):
        assert(not next(model.parameters()).is_cuda)
        if isinstance(child, CDBN2d):
            m = nn.BatchNorm2d(child.num_features)
            if use_target:
                m.load_state_dict(child.BN_T.state_dict())
            else:
                m.load_state_dict(child.BN_S.state_dict())
            setattr(model, child_name, m)
        elif isinstance(child, CDBN1d):
            m = nn.BatchNorm1d(child.num_features)
            if use_target:
                m.load_state_dict(child.BN_T.state_dict())
            else:
                m.load_state_dict(child.BN_S.state_dict())
            setattr(model, child_name, m)
        else:
            convert_bn(child, use_target=use_target)
