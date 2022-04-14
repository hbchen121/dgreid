from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta

from sklearn.cluster import DBSCAN

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.append(".")
from reid import datasets
from reid import models
from reid.models.dsbn import convert_dsbn
from reid.models.csbn import convert_csbn
from reid.layers.xbm import XBM
from reid.trainers import UDA_Baseline_Trainer
from reid.evaluation_metrics.evaluators import Evaluator, extract_features
from reid.utils.data import IterLoader
from reid.utils.data import transforms as T
from reid.utils.data.sampler import RandomMultipleGallerySampler
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid.utils.rerank import compute_jaccard_distance

from reid.config import get_parser
from reid.utils.data import CommDataset
from reid.layers import batch_norm

start_epoch = best_mAP = 0

def get_data(name, data_dir):
    data_dir = '/data/datasets'
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
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
	         T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
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
    model = models.create(args.arch, num_features=args.features, norm=False, dropout=args.dropout, 
                          num_classes=args.nclass, args=args)

    if args.csdn:
        convert_csbn(model)
    else:
        convert_dsbn(model)

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
    dataset_source = get_data(args.dataset_source, args.data_dir)
    dataset_source = CommDataset(dataset_source.train)
    print("==> Load target-domain dataset")
    dataset_target_test = get_data(args.dataset_target, args.data_dir)
    dataset_target = CommDataset(dataset_target_test.train)
    test_loader_target = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers)
    test_loader_target_test = get_test_loader(dataset_target_test, args.height, args.width, args.batch_size, args.workers)
    train_loader_source = get_train_loader(args, dataset_source, args.height, args.width,
                                           args.batch_size, args.workers, args.num_instances, iters)

    source_classes = dataset_source.num_train_pids

    args.nclass = source_classes+len(dataset_target.train)
    args.s_class = source_classes
    args.t_class = len(dataset_target.train)

    # Create model
    model = create_model(args)
    print(model)

    # Create XBM
    
    datasetSize = len(dataset_source.train)+len(dataset_target.train)

    args.memorySize = int(args.ratio*datasetSize)
    xbm = XBM(args.memorySize, args.featureSize)
    print('XBM memory size = ', args.memorySize)
    # Initialize source-domain class centroids
    sour_cluster_loader = get_test_loader(dataset_source, args.height, args.width,
                                    args.batch_size, args.workers, testset=sorted(dataset_source.train))
    source_features, _ = extract_features(model, sour_cluster_loader, print_freq=50)
    sour_fea_dict = collections.defaultdict(list)
    for f, pid, _ in sorted(dataset_source.train):
        sour_fea_dict[pid].append(source_features[f].unsqueeze(0))
    source_centers = [torch.cat(sour_fea_dict[pid],0).mean(0) for pid in sorted(sour_fea_dict.keys())]
    source_centers = torch.stack(source_centers,0)
    source_centers = F.normalize(source_centers, dim=1)
    model.module.classifier.weight.data[0:source_classes].copy_(source_centers.cuda())

    del source_centers, sour_cluster_loader, sour_fea_dict

    # Evaluator
    evaluator = Evaluator(model)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Trainer
    trainer = UDA_Baseline_Trainer(args, model, xbm, args.nclass)

    for epoch in range(args.epochs):

        tgt_cluster_loader = get_test_loader(dataset_target, args.height, args.width,
                                             args.batch_size, args.workers, testset=sorted(dataset_target.train))
        target_features, _ = extract_features(model, tgt_cluster_loader, print_freq=50)
        target_features = torch.cat([target_features[f].unsqueeze(0) for f, _, _ in
                                     sorted(dataset_target.train)], 0)

        del tgt_cluster_loader
        print('==> Create pseudo labels for unlabeled target domain with DBSCAN clustering')
      
        rerank_dist = compute_jaccard_distance(target_features, k1=args.k1, k2=args.k2, use_gpu=False).numpy()
        print('Clustering and labeling...')
        eps = args.eps
        cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
        labels = cluster.fit_predict(rerank_dist)
        num_ids = len(set(labels)) - (1 if -1 in labels else 0)
        args.t_class = num_ids

        print('\n Clustered into {} classes \n'.format(args.t_class))


        # generate new dataset and calculate cluster centers
        new_dataset = []
        cluster_centers = collections.defaultdict(list)
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_target.train), labels)):
            if label == -1: continue
            new_dataset.append((fname, source_classes+label, cid))
            cluster_centers[label].append(target_features[i])

        cluster_centers = [torch.stack(cluster_centers[idx]).mean(0) for idx in sorted(cluster_centers.keys())]
        cluster_centers = torch.stack(cluster_centers)
        model.module.classifier.weight.data[args.s_class:args.s_class+args.t_class].copy_(F.normalize(cluster_centers, dim=1).float().cuda())
    
        del cluster_centers, target_features

        train_loader_target = get_train_loader(args, dataset_target, args.height, args.width,
                                               args.batch_size, args.workers, args.num_instances, iters,
                                               trainset=new_dataset)

        train_loader_source.new_epoch()
        train_loader_target.new_epoch()
        trainer.train(epoch, train_loader_source, train_loader_target, args.s_class, args.t_class, optimizer, 
                      print_freq=args.print_freq, train_iters=args.iters, use_xbm=args.use_xbm)
                      
        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):

            print('Test on target: ', args.dataset_target)
            _, mAP = evaluator.evaluate(test_loader_target_test, dataset_target_test.query, dataset_target_test.gallery, cmc_flag=True)
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))
            print(' * Current method log dir {}\n'.
                  format(args.logs_dir))

        lr_scheduler.step()


    print ('==> Test with the best model on the target domain:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader_target_test, dataset_target_test.query, dataset_target_test.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on UDA re-ID")

    # data
    parser.add_argument('-ds', '--dataset-source', type=str, default='dukemtmc')
    parser.add_argument('-dt', '--dataset-target', type=str, default='market1501')
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--nclass', type=int, default=1000,
                        help="number of classes (source+target)")
    parser.add_argument('--s-class', type=int, default=1000,
                        help="number of classes (source)")
    parser.add_argument('--t-class', type=int, default=1000,
                        help="number of classes (target)")
    # loss
    parser.add_argument('--margin', type=float, default=0.3,
                        help="margin for triplet loss")
    parser.add_argument('--cls-scale', type=float, default=32,
                        help="scale for margin-based softmax loss")
    parser.add_argument('--cls-margin', type=float, default=0.35,
                        help="margin for margin-based softmax loss")
    parser.add_argument('--mu1', type=float, default=0.5,
                        help="weight for loss_bridge_pred")
    parser.add_argument('--mu2', type=float, default=0.1,
                        help="weight for loss_bridge_feat")
    parser.add_argument('--mu3', type=float, default=1,
                        help="weight for loss_div")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50_idm',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('-ibn', '--with-ibn', action='store_true',
                        help="if True: backbone with IBN")
    parser.add_argument('--bn-type', type=str, default='BN',
                        choices=batch_norm.bn_type)
    parser.add_argument('-se', '--with-se', action='store_true',
                        help="if True: backbone with SE block")
    parser.add_argument('-nl', '--with-nl', action='store_true',
                        help="if True: backbone with Non local")
    parser.add_argument('-in', '--with-in', action='store_true',
                        help="if True: backbone with instance norm after bottleneck")
    parser.add_argument('-srm', '--with-srm', action='store_true',
                        help="if True: backbone with SRM block")
    parser.add_argument('-csrm', '--with-csrm', action='store_true',
                        help="if True: backbone with CSRM block")
    parser.add_argument('-smm', '--with-smm', action='store_true',
                        help="if True: backbone with Style Mixture Module(SMM)")
    parser.add_argument('--smm-half', action='store_true',
                        help="if True: half mix in SMM")
    parser.add_argument('--smm-in', action='store_true',
                        help="if True: in in SMM")
    parser.add_argument('--smm-stage', type=int, default=1,
                        help="smm plug-in stage in backbone", choices=[-1, 0, 1, 2, 3, 4])
    parser.add_argument('--smm-lam', type=float, default=1.,
                        help="mixture lambda for smm")

    # xbm parameters
    parser.add_argument('--memorySize', type=int, default=8192,
                        help='meomory bank size')
    parser.add_argument('--ratio', type=float, default=1,
                        help='memorySize=ratio*data_size')
    parser.add_argument('--featureSize', type=int, default=2048)
    parser.add_argument('--use-xbm', action='store_true',help="if True: strong baseline; if False: naive baseline")

    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=30)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--eval-step', type=int, default=10)

    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))

    # hbchen
    parser.add_argument('--csdn', type=bool, default=False)
    main()

