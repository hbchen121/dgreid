from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import sys
sys.path.append(".")

from reid import models
from reid.layers import softmaxs
from reid.layers import batch_norm



def get_parser(description="Baseline for DG Person re-ID"):
    parser = argparse.ArgumentParser(description=description)
    # data
    parser.add_argument('-ds', '--dataset-source', type=str, default='dukemtmc')
    parser.add_argument('-dt', '--dataset-target', type=str, default='market1501')
    parser.add_argument('--combine-all', action='store_true',
                        help="if True: combinall train, query, gallery for training;")
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # loss
    parser.add_argument('--margin', type=float, default=0.3,
                        help="margin for triplet loss")
    parser.add_argument('--cls-scale', type=float, default=32,
                        help="scale for margin-based softmax loss")
    parser.add_argument('--cls-margin', type=float, default=0.35,
                        help="margin for margin-based softmax loss")
    parser.add_argument('--loss-diff', action='store_true',
                        help="if True: back propagation difference loss")
    parser.add_argument('--global-lambda', type=float, default=0.,
                        help="lambda weight of global classifier and triplet loss")
    parser.add_argument('--local-lambda', type=float, default=1.,
                            help="lambda weight of local classifier and triplet loss")

    # model.backbone
    parser.add_argument('-a', '--arch', type=str, default='resnet50_mde',
                        choices=models.names())
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
    parser.add_argument('--smm-stage', type=str, default="1",
                        help="smm plug-in stage in backbone")
    parser.add_argument('--smm-lam', type=float, default=1.,
                        help="mixture lambda for smm")
    parser.add_argument('--smm-half-id', action='store_true',
                        help="if True: half (spilt) the samples with the same identity")
    parser.add_argument('--smm-mix-layer', type=str, default='AdaIN',
                        help="style mix layer in smm", choices=['AdaIN', 'AdaAttN'])
    parser.add_argument('--smm-lambda-content', type=float, default=0.,
                        help="weight of content loss")
    parser.add_argument('--smm-lambda-style', type=float, default=1.,
                        help="weight of style loss")
    parser.add_argument('--smm-scale', type=str, default="",
                        help="smm scale")


    # model.heads
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('-cls', '--classifier', action='store_true',
                        help="if True: using linear classifiers;")
    parser.add_argument('--cls-type', type=str, default='Linear',
                        choices=softmaxs.__all__)
    parser.add_argument('--cls-weight', action='store_true',
                        help="if True: init linear-cls weight by features")

    # memory cls parameters
    parser.add_argument('-mem', '--with-memory', action='store_true',
                        help="using memory classifiers")
    parser.add_argument('--mem-pairs', action='store_true',
                        help="if True: using pairs memory;")
    parser.add_argument('--momentum', type=float, default=0.1,
                        help="update momentum for the hybrid memory")
    parser.add_argument('--dynamic-momentum', type=int, default=0,
                        help="base of dynamic step")
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    parser.add_argument('--mem-type', type=str, default='cos',
                        choices=['cos', 'arc', 'circle'])
    parser.add_argument('--mem-margin', type=float, default=0.0,
                        help="margin for memory classifier")
    parser.add_argument('--mean-update', action='store_true',
                        help="if True: update each pid feature of memory by its features's mean")


    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--step-size', type=int, default=40)

    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--eval-step', type=int, default=10)

    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, default='/data/datasets')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    return parser
