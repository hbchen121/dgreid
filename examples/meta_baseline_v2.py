from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta
import tabulate
from termcolor import colored

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.append(".")
from reid import datasets
from reid import models
from reid.config import get_parser
from reid.layers import softmaxs
from reid.layers.mix_memory import MixtureDomainMemory
from reid.trainers.meta_trainers_v2 import MDE_Trainer
from reid.evaluation_metrics.evaluators import Evaluator, extract_features
from reid.utils.data import CommDataset
from reid.utils.data import IterLoader
from reid.utils.data import transforms as T
from reid.utils.data.sampler import RandomMultipleGallerySampler
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
from torch.utils.tensorboard import SummaryWriter
from reid.layers.classifier import build_mde_classifier
from reid.solver import WarmupMultiStepLR
from reid.utils.relabel import ReLabel

start_epoch = best_mAP = 0

def get_data(name, data_dir, combineall=False):
    # data_dir = '/data/datasets'
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root, combineall=combineall)
    return dataset

def get_train_loader(args, dataset, height, width, batch_size, workers,
                    num_instances, iters, trainset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
	         # T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
                DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=args.nclass, args=args)
    def inplace_relu(m):
        classname = m.__class__.__name__
        if classname.find('ReLU') != -1:
            m.inplace = True

    model.apply(inplace_relu)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters>0) else None
    print("==> Load source-domain dataset")
    dataset_sources = []
    source_nclass = []
    for src in args.dataset_source.split(','):
        dataset = get_data(src, args.data_dir, args.combine_all)
        dataset_source = CommDataset(dataset.train, verbose=False)
        dataset_sources.append(dataset_source)
        source_nclass.append(dataset_source.num_train_pids)

    args.nsource = len(source_nclass)
    args.nclass = source_nclass
    relabel = ReLabel(source_nclass)
    args.relabel = relabel
    args.global_classifier = args.global_lambda > 0 and args.classifier
    args.local_classifiers = args.local_lambda > 0 and args.classifier

    args.global_memory = args.global_lambda > 0 and args.with_memory
    args.local_memory = args.local_lambda > 0 and args.with_memory

    args.smm_stage = [int(stage) for stage in args.smm_stage.split(',')]
    args.smm_stage.sort()
    assert (False not in [stage in [-1, 0, 1, 2, 3, 4] for stage in args.smm_stage])

    if args.smm_scale == "":
        args.smm_scale = None
    else:
        smm_scales = args.smm_scale.split(',')
        args.smm_scale = []
        scale_len = 0
        for smm_scale in smm_scales:
            smm_scale = [int(scale) for scale in smm_scale.split('_')]
            assert (False not in [(scale == 0 or 64 % scale == 0) for scale in smm_scale])
            scale_len = max(len(smm_scale), scale_len)
            args.smm_scale.append(smm_scale)
        args.scale_len = scale_len

    # print(args.smm_scale)
    # assert False

    # args.smm_stage = [int(stage) for stage in args.smm_stage.split(',')]
    # assert (False not in [stage in [-1, 0, 1, 2, 3, 4] for stage in args.smm_stage])

    print("==> Load target-domain dataset")
    # dataset_target = get_data(args.dataset_target, args.data_dir)
    target_loaders = []
    target_datasets = []
    target_dataset_names = args.dataset_target.split(',')
    for target_dataset_name in target_dataset_names:
        target_dataset = get_data(target_dataset_name, args.data_dir)
        target_loader = get_test_loader(target_dataset, args.height, args.width, args.batch_size, args.workers)
        target_loaders.append(target_loader)
        target_datasets.append(target_dataset)

    # DataLoaders
    # test_loader_target = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers)
    train_loader_sources = []
    for dataset_source in dataset_sources:
        train_loader_source = get_train_loader(args, dataset_source, args.height, args.width,
                                           args.batch_size, args.workers, args.num_instances, iters)
        train_loader_sources.append(train_loader_source)

    # Create model
    model = create_model(args)
    print(model)

    # Evaluator
    evaluator = Evaluator(model)
    best_mAP = [0] * len(target_datasets)

    # Memory classifier
    if args.with_memory:
        mixMem = MixtureDomainMemory(model.module.num_features, args.nsource, args.nclass, temp=args.temp,
                                     momentum=args.momentum, margin=args.mem_margin,
                                     num_instances=args.num_instances,
                                     dynamic_momentum=args.dynamic_momentum).cuda()
        sour_fea_dict = collections.defaultdict(list)

        for i in range(args.nsource):
            dataset_source = dataset_sources[i]
            sour_cluster_loader = get_test_loader(dataset_source, args.height, args.width,
                                                  args.batch_size, args.workers, testset=dataset_source.train)
            source_features, _ = extract_features(model, sour_cluster_loader, print_freq=50)
            for f, pid, _ in dataset_source.train:
                pid_new = relabel(i, pid).item()
                sour_fea_dict[pid_new].append(source_features[f].unsqueeze(0))

        source_centers = [torch.cat(sour_fea_dict[pid], 0).mean(0) for pid in sour_fea_dict.keys()]
        source_centers = torch.stack(source_centers, 0)  ## pid,2048
        source_centers = F.normalize(source_centers, dim=1).cuda()

        mixMem.init(source_centers, torch.arange(relabel.idxSum[-1]).cuda())
        mixMem = nn.DataParallel(mixMem)
        print(mixMem)
        del source_centers, sour_cluster_loader, sour_fea_dict
        args.memory = mixMem

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    # lr_scheduler = WarmupMultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.1, warmup_factor=0.01,
    #                                  warmup_iters=10, warmup_method="linear")

    # Tensorboard Log
    writer = SummaryWriter(log_dir=args.logs_dir)
    args.writer = writer

    # Trainer

    trainer = MDE_Trainer(args, model, source_nclass)

    table = []
    # header = ['Epoch', 'mAP', 'Rank-1', 'Rank-5', 'Rank-10']
    header = ['Epoch', 'Dataset', 'mAP', 'Rank-1', 'Rank-5', 'Rank-10']
    table.append(header)

    for epoch in range(args.epochs):

        if not args.combine_all:
            for train_loader_source in train_loader_sources:
                train_loader_source.new_epoch()

        trainer.train(epoch, train_loader_sources, optimizer, print_freq=args.print_freq, train_iters=args.iters)

        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            for target_id in range(len(target_datasets)):
                target_dataset_name = target_dataset_names[target_id]
                target_dataset = target_datasets[target_id]
                target_loader = target_loaders[target_id]
                print('Test on target: ', target_dataset_name)
                result_dict, mAP = evaluator.evaluate(target_loader, target_dataset.query, target_dataset.gallery,
                                                      cmc_flag=True)

                # show results in table
                record = list()
                record.append(epoch)
                record.append(target_dataset_name)
                record.append(result_dict['mAP'])
                record.append(result_dict['rank-1'])
                record.append(result_dict['rank-5'])
                record.append(result_dict['rank-10'])
                table.append(record)

                print(tabulate.tabulate(table, headers='firstrow', tablefmt='github', floatfmt='.2%'))

                is_best = mAP > best_mAP[target_id]
                best_mAP[target_id] = max(mAP, best_mAP[target_id])

                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'best_mAP': best_mAP,
                }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

                print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                      format(epoch, mAP, best_mAP[target_id], ' *' if is_best else ''))
                print(' * Current method log dir {}\n'.
                      format(args.logs_dir))
    #
    #     if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
    #         print('Test on target: ', args.dataset_target)
    #         result_dict, mAP = evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)
    #
    #         # show results in table
    #         record = []
    #         record.append(epoch)
    #         record.append(result_dict['mAP'])
    #         record.append(result_dict['rank-1'])
    #         record.append(result_dict['rank-5'])
    #         record.append(result_dict['rank-10'])
    #         table.append(record)
    #
    #         ttable = tabulate.tabulate(table, headers='firstrow', tablefmt='github', floatfmt='.2%', numalign="left")
    #
    #         print(f"=> All results in csv format: \n" + colored(ttable, "cyan"))
    #
    #         is_best = (mAP > best_mAP)
    #         best_mAP = max(mAP, best_mAP)
    #         save_checkpoint({
    #             'state_dict': model.state_dict(),
    #             'epoch': epoch + 1,
    #             'best_mAP': best_mAP,
    #         }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))
    #
    #         print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
    #               format(epoch, mAP, best_mAP, ' *' if is_best else ''))
    #         print(' * Current method log dir {}\n'.
    #               format(args.logs_dir))
    #
        lr_scheduler.step()

    args.writer.flush()

    # print ('==> Test with the best model on the target domain:')
    # checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    # model.load_state_dict(checkpoint['state_dict'])
    # evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = get_parser(description="Multi-Domain Equality (MDE) Baseline for Person re-ID")

    parser.add_argument('--meta-beta', type=float, default=0.5,
                        help="")
    main()

