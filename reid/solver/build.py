# encoding: utf-8

import torch
from .lr_scheduler import WarmupMultiStepLR


def make_optimizer(args, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.lr
        weight_decay = args.weight_decay
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = getattr(torch.optim, 'Adam')(params)
    return optimizer


def get_default_optimizer_scheduler(args, model):
    params = [
        {"params": [p for _, p in model.named_parameters() if p.requires_grad]},
    ]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones=[30, 50, 70], gamma=0.1, warmup_factor=0.01,
                                     warmup_iters=10, warmup_method="linear")
    return optimizer, lr_scheduler
