from __future__ import print_function, absolute_import
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from ..utils.meters import AverageMeter
from ..evaluation_metrics import accuracy
from ..loss import TripletLoss, CrossEntropyLabelSmooth
from .trainers import TrainerBase
from ..loss.triplet import euclidean_dist, cosine_dist
from ..utils.tools import write_json
import os.path as osp
import numpy as np
from reid.models.MetaModules import MixUpBatchNorm1d as MixUp1D
from reid.models import create


def vector_cosine(x, y):
    dist = cosine_dist(x, y)
    dist = torch.diag(dist)
    return dist.sum()


class META_Trainer(TrainerBase):
    """
    MDE DG Trainer, data_loader_source has multiple data loader
    """
    def __init__(self, args, model, n_num_classes):
        super(META_Trainer, self).__init__(args, model)
        self.model = model
        self.n_source = len(n_num_classes)
        self.n_num_classes = n_num_classes
        self.classifiers = args.classifier
        self.criterion_ce = CrossEntropyLabelSmooth().cuda()
        self.criterion_tri = TripletLoss(margin=args.margin).cuda()

    def _forward(self, inputs):
        bn_x, x = self.model(inputs)
        bn_x = torch.chunk(bn_x, self.n_source, dim=0)
        probs = []
        for i in range(self.n_source):
            prob = self.classifiers[i](bn_x[i])
            probs.append(prob)
        return probs, x

    def train(self, epoch, data_loader_sources, optimizer, print_freq=50, train_iters=400):

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        # self.weight_record(epoch)

        for i in range(train_iters):
            self.iter = epoch * train_iters + i
            self.writer.add_scalar('epoch', epoch, self.iter)
            print_now = False  # (i + 1) % (4 * print_freq) == 0

            # load data
            n_source_inputs = []
            for data_loader_source in data_loader_sources:
                source_inputs = data_loader_source.next()

                n_source_inputs.append(source_inputs)
            # target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            n_inputs = []
            n_targets = []
            for source_inputs in n_source_inputs:
                inputs, targets, _ = self._parse_data(source_inputs)
                n_inputs.append(inputs)
                n_targets.append(targets)

            inputs = torch.cat(tuple(n_inputs), dim=0)  # .contiguous()

            # forward
            probs, feats = self._forward(inputs)

            n_feats = torch.chunk(feats, self.n_source, dim=0)

            # classification+triplet
            loss_ce = 0.
            loss_tri = 0.
            for j in range(self.n_source):
                ce = self.criterion_ce(probs[j], n_targets[j])
                loss_tri += self.criterion_tri(n_feats[j], n_targets[j])
                loss_ce += ce
                if print_now:
                    print(str(j) + ': ' + str(ce.item()))
                # loss += loss_ce + loss_tri

            if print_now:
                print('-' * 25)

            loss_ce /= self.n_source
            loss_tri /= self.n_source
            loss = loss_ce + loss_tri

            optimizer.zero_grad()
            loss.backward()
            self.printGrad(self.classifiers)
            # import IPython
            # IPython.embed()
            optimizer.step()
            precs = 0.
            for j in range(self.n_source):
                # import IPython
                # IPython.embed()
                prec, = accuracy(probs[j].view(-1, probs[j].size(-1)).data, n_targets[j].data)
                if print_now:
                    print(str(j) + ': ' + str(prec[0].item()))
                precs += prec[0]
            if print_now:
                print('-' * 25)
            precs /= self.n_source

            losses.update(loss.item())
            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            precisions.update(precs)

            # add tensorboard
            self.writer.add_scalar('loss', loss.item(), self.iter)
            self.writer.add_scalar('loss_ce', loss_ce.item(), self.iter)
            self.writer.add_scalar('loss_tri', loss_tri.item(), self.iter)
            self.writer.add_scalar('Prec', precs, self.iter)

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:

                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f}) '
                      'Data {:.3f} ({:.3f}) '
                      'Loss {:.3f} ({:.3f}) '
                      'Loss_ce {:.3f} ({:.3f}) '
                      'Loss_tri {:.3f} ({:.3f}) '
                      'Prec_s {:.2%} ({:.2%}) '
                      'lr {:.3e}'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tri.val, losses_tri.avg,
                              precisions.val, precisions.avg,
                              self.get_lr(optimizer),
                              ))


class META_MC_Trainer(META_Trainer):
    """
    MDE DG Trainer, data_loader_source has multiple data loader, with memory classifier
    """
    def __init__(self, args, model, n_num_classes, memories):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.memories = memories
        super(META_MC_Trainer, self).__init__(args, model, n_num_classes)
        # self.classifiers = None

    def _forward(self, inputs):
        bn_x, x, norm_feats = self.model(inputs)
        bn_x = torch.chunk(bn_x, self.n_source, dim=0)
        probs = []
        if self.classifiers:
            for i in range(self.n_source):
                prob = self.classifiers[i](bn_x[i])
                probs.append(prob)
        return probs, x, norm_feats

    def train2(self, epoch, data_loader_sources, optimizer, print_freq=50, train_iters=200):

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_meta_train = AverageMeter()
        losses_meta_test = AverageMeter()

        metaLR =  self.get_lr(optimizer)

        end = time.time()

        for i in range(train_iters):

            self.iter = epoch * train_iters + i
            self.writer.add_scalar('epoch', epoch, self.iter)
            self.writer.add_scalar('lr', self.get_lr(optimizer), self.iter)

            if self.args.arch == 'resMetaMix':
                network_bns = [x for x in list(self.model.modules()) if isinstance(x, MixUp1D)]

                for bn in network_bns:
                    bn.meta_mean1 = torch.zeros(bn.meta_mean1.size()).float().cuda()
                    bn.meta_var1 = torch.zeros(bn.meta_var1.size()).float().cuda()
                    bn.meta_mean2 = torch.zeros(bn.meta_mean2.size()).float().cuda()
                    bn.meta_var2 = torch.zeros(bn.meta_var2.size()).float().cuda()

            # load data
            test_idx = np.random.choice(self.n_source)
            train_idxs = [i for i in range(self.n_source)]
            del train_idxs[test_idx]
            n_source_inputs = []
            for data_loader_source in data_loader_sources:
                source_inputs = data_loader_source.next()
                n_source_inputs.append(source_inputs)
            data_time.update(time.time() - end)

            # process inputs
            # meta test
            test_inputs, test_pids, _ = self._parse_data(n_source_inputs[test_idx])

            loss_meta_train = 0.
            save_index = 0
            for idx in train_idxs:
                train_inputs = n_source_inputs[idx]
                inputs, targets, _ = self._parse_data(train_inputs)
                save_index += 1
                f_out, tri_features = self.model(inputs, MTE='', save_index=save_index)
                loss_mtr_tri = self.criterion_tri(tri_features, targets)
                loss_s = self.memories[idx](f_out, targets).mean()

                loss_meta_train = loss_meta_train + loss_s + loss_mtr_tri

            loss_meta_train = loss_meta_train / (self.n_source - 1)


            self.model.zero_grad()
            # optimizer.zero_grad()
            grad_info = torch.autograd.grad(loss_meta_train, self.model.module.params(), create_graph=True)
            # grad_info = torch.autograd.grad(loss_meta_train, self.model.module.parameters(), create_graph=True)
            self.newMeta = create(self.args.arch, norm=True, BNNeck=self.args.BNNeck)
            self.newMeta.copyModel(self.model.module)
            self.newMeta.update_params(
                lr_inner=metaLR, source_params=grad_info, solver='adam'
            )
            del grad_info

            self.newMeta = nn.DataParallel(self.newMeta).to(self.device)

            f_test, mte_tri_feats = self.newMeta(test_inputs, MTE=self.args.BNtype)

            loss_meta_test = 0.
            if isinstance(f_test, list):
                for feature in f_test:
                    loss_meta_test += self.memories[test_idx](feature, test_pids).mean()
                loss_meta_test /= len(f_test)

            else:
                loss_meta_test = self.memories[test_idx](f_test, test_pids).mean()

            loss_mte_tri = self.criterion_tri(mte_tri_feats, test_pids)
            loss_meta_test = loss_meta_test + loss_mte_tri

            loss_final = loss_meta_train + loss_meta_test
            losses_meta_train.update(loss_meta_train.item())
            losses_meta_test.update(loss_meta_test.item())
            losses.update(loss_final.item())

            optimizer.zero_grad()
            loss_final.backward()
            optimizer.step()

            # update memory classifiers
            with torch.no_grad():
                for j in range(self.n_source):
                    imgs, pids, _ = self._parse_data(n_source_inputs[j])
                    f_new, _ = self.model(imgs)
                    self.memories[j].module.MomentumUpdate(f_new, pids)

            # add tensorboard
            self.writer.add_scalar('loss', loss_final.item(), self.iter)
            self.writer.add_scalar('loss_mtr', loss_meta_train.item(), self.iter)
            self.writer.add_scalar('loss_mte', loss_meta_test.item(), self.iter)

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:

                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f}) '
                      'Loss {:.3f} ({:.3f}) '
                      'Loss_mtr {:.3f} ({:.3f}) '
                      'Loss_mte {:.3f} ({:.3f}) '
                      'lr {:.3e} '
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              losses.val, losses.avg,
                              losses_meta_train.val, losses_meta_train.avg,
                              losses_meta_test.val, losses_meta_test.avg,
                              self.get_lr(optimizer),
                              ))

    def train(self, epoch, data_loader_sources, optimizer, print_freq=50, train_iters=200):
        """MetaResNet """

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_meta_train = AverageMeter()
        losses_meta_test = AverageMeter()

        metaLR =  self.get_lr(optimizer)

        end = time.time()

        for i in range(train_iters):

            self.iter = epoch * train_iters + i
            self.writer.add_scalar('epoch', epoch, self.iter)
            self.writer.add_scalar('lr', self.get_lr(optimizer), self.iter)

            assert self.args.arch == 'resMeta'

            # load data
            test_idx = np.random.choice(self.n_source)
            train_idxs = [i for i in range(self.n_source)]
            del train_idxs[test_idx]
            n_source_inputs = []
            for data_loader_source in data_loader_sources:
                source_inputs = data_loader_source.next()
                n_source_inputs.append(source_inputs)
            data_time.update(time.time() - end)

            # process inputs
            # meta test
            test_inputs, test_pids, _ = self._parse_data(n_source_inputs[test_idx])

            n_inputs = []
            n_targets = []
            for idx in train_idxs:
                train_inputs = n_source_inputs[idx]
                inputs, targets, _ = self._parse_data(train_inputs)
                n_inputs.append(inputs)
                n_targets.append(targets)
            inputs = torch.cat(tuple(n_inputs), dim=0).contiguous()

            # meta train
            f_out, tri_features = self.model(inputs)
            n_f_out = torch.chunk(f_out, self.n_source - 1, dim=0)
            n_tri_features = torch.chunk(tri_features, self.n_source - 1, dim=0)

            loss_meta_train = 0.
            for j, idx in enumerate(train_idxs):
                loss_mtr_tri = self.criterion_tri(n_tri_features[j], n_targets[j])
                loss_s = self.memories[idx](n_f_out[j], n_targets[j]).mean()
                loss_meta_train += loss_meta_train + loss_s + loss_mtr_tri

            loss_meta_train = loss_meta_train / (self.n_source - 1)

            self.model.zero_grad()
            # optimizer.zero_grad()
            grad_info = torch.autograd.grad(loss_meta_train, self.model.module.params(), create_graph=True)
            self.newMeta = create(self.args.arch, norm=True, BNNeck=self.args.BNNeck)
            self.newMeta.copyModel(self.model.module)
            self.newMeta.update_params(
                lr_inner=metaLR, source_params=grad_info, solver='adam'
            )
            del grad_info

            self.newMeta = nn.DataParallel(self.newMeta).to(self.device)

            f_test, mte_tri_feats = self.newMeta(test_inputs, MTE=self.args.BNtype)

            loss_meta_test = 0.
            if isinstance(f_test, list):
                for feature in f_test:
                    loss_meta_test += self.memories[test_idx](feature, test_pids).mean()
                loss_meta_test /= len(f_test)

            else:
                loss_meta_test = self.memories[test_idx](f_test, test_pids).mean()

            loss_mte_tri = self.criterion_tri(mte_tri_feats, test_pids)
            loss_meta_test = loss_meta_test + loss_mte_tri

            loss_final = loss_meta_train + loss_meta_test
            losses_meta_train.update(loss_meta_train.item())
            losses_meta_test.update(loss_meta_test.item())
            losses.update(loss_final.item())

            optimizer.zero_grad()
            loss_final.backward()
            optimizer.step()

            # update memory classifiers
            with torch.no_grad():
                for j in range(self.n_source):
                    imgs, pids, _ = self._parse_data(n_source_inputs[j])
                    f_new, _ = self.model(imgs)
                    self.memories[j].module.MomentumUpdate(f_new, pids)

            # add tensorboard
            self.writer.add_scalar('loss', loss_final.item(), self.iter)
            self.writer.add_scalar('loss_mtr', loss_meta_train.item(), self.iter)
            self.writer.add_scalar('loss_mte', loss_meta_test.item(), self.iter)

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:

                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f}) '
                      'Loss {:.3f} ({:.3f}) '
                      'Loss_mtr {:.3f} ({:.3f}) '
                      'Loss_mte {:.3f} ({:.3f}) '
                      'lr {:.3e} '
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              losses.val, losses.avg,
                              losses_meta_train.val, losses_meta_train.avg,
                              losses_meta_test.val, losses_meta_test.avg,
                              self.get_lr(optimizer),
                              ))

    def train_MD(self, epoch, data_loader_sources, optimizer, print_freq=50, train_iters=200):

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_meta_train = AverageMeter()
        losses_meta_test = AverageMeter()

        metaLR = self.get_lr(optimizer)

        end = time.time()

        for i in range(train_iters):

            self.iter = epoch * train_iters + i
            self.writer.add_scalar('epoch', epoch, self.iter)
            self.writer.add_scalar('lr', self.get_lr(optimizer), self.iter)

            for train_idx in range(self.n_source):
                """MD-ExCo 中对每一训练集进行 meta learning"""
                # load data
                test_idxs = [idx for idx in range(self.n_source)]
                del test_idxs[train_idx]
                test_idx = np.random.choice(test_idxs)

                # load data
                train_inputs_loader = data_loader_sources[train_idx].next()
                test_inputs_loader = data_loader_sources[test_idx].next()
                data_time.update(time.time() - end)

                # meta train
                inputs, targets, _ = self._parse_data(train_inputs_loader)
                f_out, tri_features = self.model(inputs)
                loss_mtr_tri = self.criterion_tri(tri_features, targets)
                loss_mtr_mem = self.memories[train_idx](f_out, targets).mean()
                loss_meta_train = loss_mtr_mem + loss_mtr_tri

                # meta model update
                self.model.zero_grad()
                # optimizer.zero_grad()
                grad_info = torch.autograd.grad(loss_meta_train, self.model.module.params(), create_graph=True)
                # grad_info = torch.autograd.grad(loss_meta_train, self.model.module.parameters(), create_graph=True)
                self.newMeta = create(self.args.arch, norm=True, BNNeck=self.args.BNNeck)
                self.newMeta.copyModel(self.model.module)
                self.newMeta.update_params(
                    lr_inner=metaLR, source_params=grad_info, solver='adam'
                )
                del grad_info

                self.newMeta = nn.DataParallel(self.newMeta).to(self.device)

                # meta test
                test_inputs, test_pids, _ = self._parse_data(test_inputs_loader)
                f_test, mte_tri_feats = self.newMeta(test_inputs, MTE=self.args.BNtype)

                loss_mte_mem = self.memories[test_idx](f_test, test_pids).mean()
                loss_mte_tri = self.criterion_tri(mte_tri_feats, test_pids)
                loss_meta_test = loss_mte_mem + loss_mte_tri

                # model update
                loss_final = loss_meta_train + loss_meta_test
                losses_meta_train.update(loss_meta_train.item())
                losses_meta_test.update(loss_meta_test.item())
                losses.update(loss_final.item())

                optimizer.zero_grad()
                loss_final.backward()
                optimizer.step()

                # update memory classifiers
                with torch.no_grad():
                    # train memory
                    f_new, _ = self.model(inputs)
                    self.memories[train_idx].module.MomentumUpdate(f_new, targets)
                # with torch.no_grad():
                #     for j in range(self.n_source):
                #         imgs, pids, _ = self._parse_data(n_source_inputs[j])
                #         f_new, _ = self.model(imgs)
                #         self.memories[j].module.MomentumUpdate(f_new, pids)

            # add tensorboard
            self.writer.add_scalar('loss', losses.val, self.iter)
            self.writer.add_scalar('loss_mtr', losses_meta_train.val, self.iter)
            self.writer.add_scalar('loss_mte', losses_meta_test.val, self.iter)

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f}) '
                      'Loss {:.3f} ({:.3f}) '
                      'Loss_mtr {:.3f} ({:.3f}) '
                      'Loss_mte {:.3f} ({:.3f}) '
                      'lr {:.3e} '
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              losses.val, losses.avg,
                              losses_meta_train.val, losses_meta_train.avg,
                              losses_meta_test.val, losses_meta_test.avg,
                              self.get_lr(optimizer),
                              ))
