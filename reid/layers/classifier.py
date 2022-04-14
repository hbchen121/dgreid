import torch
import torch.nn.functional as F
from torch.nn import init
from torch import nn
from reid.layers import softmaxs


def build_domain_classifier(args):
    num_features = args.features if args.features > 0 else 2048
    nums_domain = args.nsource
    assert nums_domain > 1, "domain nums is too less, only: {}".format(str(nums_domain))
    classifier = softmaxs.Linear(num_features, nums_domain)
    return classifier


def build_mde_classifier(args):
    num_features = args.features if args.features > 0 else 2048
    assert hasattr(softmaxs, args.cls_type), "Expected cls types are {}, " \
                                               "but got {}".format(softmaxs.__all__, args.cls_type)
    classifiers = nn.ModuleList()
    for i, num_class in enumerate(args.nclass):
        classifier = getattr(softmaxs, args.cls_type)(num_features, num_class, args.cls_scale, args.cls_margin)
        if args.classifier_weight:
            with torch.no_grad():
                classifier.weight.data = args.classifier_weight[i]
            print("init classifier by features")
        classifiers.append(classifier)
    return classifiers


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, dim_out)
        self.linear2 = nn.Linear(dim_out, dim_out)
        # init.kaiming_normal_(self.linear1.weight, mode='fan_out')
        nn.init.normal_(self.linear1.weight, 0, 0.01)
        init.constant_(self.linear1.bias, 0)
        # init.kaiming_normal_(self.linear2.weight, mode='fan_out')
        nn.init.normal_(self.linear2.weight, 0, 0.01)
        init.constant_(self.linear2.bias, 0)

    def forward(self, features):
        return self.linear2(F.relu(self.linear1(features)))


class MLP_norm(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.W1 = nn.Parameter(torch.Tensor(dim_out, dim_in))
        self.W2 = nn.Parameter(torch.Tensor(dim_out, dim_out))
        init.normal_(self.W1, std=0.001)
        init.normal_(self.W2, std=0.001)

    def forward(self, features):
        wh = F.linear(features, self.W1)
        wh = F.relu(wh)
        return F.linear(wh, self.W2)
