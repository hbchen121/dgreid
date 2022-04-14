from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import torch
import torch.nn.functional as F
import numpy as np
# from .evaluation_metrics import cmc, mean_ap
from reid.utils.meters import AverageMeter
from reid.utils.rerank import re_ranking
from reid.evaluation_metrics.rank import evaluate_rank
from reid.utils import to_torch
from tabulate import tabulate
from termcolor import colored
from .evaluators import pairwise_distance, evaluate_all


def extract_cnn_feature(model, inputs):
    inputs = to_torch(inputs).cuda()
    outputs = model(inputs)
    # outputs = outputs.data.cpu()
    return outputs


def extract_features(model, mlp, mlp_g, data_loader, print_freq=50, evaluate=False, bnneck=None, feat_g_identity=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()
    g_features = OrderedDict()

    end = time.time()

    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            outputs_h = extract_cnn_feature(model, imgs)
            if evaluate:
                # evaluate 使用 feat_h 和 feat_z
                outputs_z = outputs_h
                outputs_z_g = mlp_g(outputs_h)
                if not feat_g_identity:
                    outputs_z_g = bnneck(outputs_z_g)
            else:
                outputs_z = mlp(outputs_h)
                outputs_z_g = mlp_g(outputs_h)
            outputs_z = F.normalize(outputs_z)
            outputs_z_g = F.normalize(outputs_z_g)
            outputs_z = outputs_z.data.cpu()
            outputs_z_g = outputs_z_g.data.cpu()
            for fname, output, output_g, pid in zip(fnames, outputs_z, outputs_z_g, pids):
                features[fname] = output
                g_features[fname] = output_g
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, labels, g_features


class Multi_Evaluator(object):
    def __init__(self, model, mlps, bnneck):
        super(Multi_Evaluator, self).__init__()
        self.model = model
        self.mlps = mlps
        self.bnneck = bnneck

    def evaluate(self, data_loader, query, gallery, cmc_flag=False, rerank=False, feat_g_identity=False):
        features_h, _, features = extract_features(self.model, self.mlps[-1], self.mlps[-1], data_loader,
                                                   evaluate=True, bnneck=self.bnneck, feat_g_identity=feat_g_identity)
        distmat, query_features, gallery_features = pairwise_distance(features, query, gallery)
        results = evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery,
                               cmc_flag=cmc_flag, method='feat_z')
        if feat_g_identity:
            results_h = (0, 0)
        else:
            distmat, query_features, gallery_features = pairwise_distance(features_h, query, gallery)
            results_h = evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery,
                                     cmc_flag=cmc_flag, method='feat_h')

        if (not rerank):
            return results, results_h

        print('Applying person re-ranking ...')
        distmat_qq, _, _ = pairwise_distance(features, query, query)
        distmat_gg, _, _ = pairwise_distance(features, gallery, gallery)
        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
        return evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)
