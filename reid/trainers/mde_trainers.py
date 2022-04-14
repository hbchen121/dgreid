from __future__ import print_function, absolute_import
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from ..utils.meters import AverageMeter
from ..evaluation_metrics import accuracy
from ..loss import TripletLoss, CrossEntropyLabelSmooth, CrossEntropy
from .trainers import TrainerBase
from ..loss.triplet import euclidean_dist, cosine_dist
from ..utils.tools import write_json
import os.path as osp


def vector_cosine(x, y):
    # dist = F.linear(F.normalize(x), F.normalize(y))
    # dist = euclidean_dist(F.normalize(x), F.normalize(y))
    # dist = F.cosine_similarity(F.normalize(x), F.normalize(y), dim=-1)
    dist = cosine_dist(x, y)
    dist = torch.diag(dist)
    return dist.sum()


def softmax_dist(w, m):
    """ w/m: classes * feat_dim (c * d)"""
    sim = F.linear(F.normalize(w), F.normalize(m))  # c * c
    labels = torch.arange(0, w.shape[0], device=w.device)  # c * 1
    loss = CrossEntropy(sim, labels)
    return loss


class MDE_Trainer(TrainerBase):
    """
    MDE DG Trainer, data_loader_source has multiple data loader
    """
    def __init__(self, args, model, n_num_classes):
        super(MDE_Trainer, self).__init__(args, model)
        self.model = model
        self.n_source = len(n_num_classes)
        self.n_num_classes = n_num_classes
        self.classifiers = args.classifier
        self.criterion_ce = CrossEntropyLabelSmooth().cuda()
        self.criterion_tri = TripletLoss(margin=args.margin).cuda()

    def printGrad(self):
        for i, cls_layer in enumerate(self.classifiers):
            with torch.no_grad():
                grad = cls_layer.weight.grad
                grad = grad.mean().data
            self.writer.add_scalar('cls_{}_gard'.format(i), grad, self.iter)

    def _forward(self, inputs, targets):
        bn_x, x = self.model(inputs)
        bn_x = torch.chunk(bn_x, self.n_source, dim=0)
        probs = []
        for i in range(self.n_source):
            prob = self.classifiers[i](bn_x[i], targets[i])
            probs.append(prob)
        return probs, x

    def weight_record(self, epoch, save_freq=10):
        if epoch == 0:
            self.weight_dict = {}
            for i, cls in enumerate(self.classifiers):
                self.weight_dict[i] = [cls.weight.cpu().data]
        else:
            for i, cls in enumerate(self.classifiers):
                self.weight_dict[i].append(cls.weight.cpu().data)
            if (epoch + 1) % save_freq == 0:
                out_dir = osp.join(self.args.logs_dir, 'weight.pt')
                torch.save(self.weight_dict, out_dir)
                print("save weight.pt at epoch {}".format(epoch))

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
            lr = self.get_lr(optimizer)
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
            probs, feats = self._forward(inputs, n_targets)

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

            if print_now:
                print('-' * 25)

            loss_ce /= self.n_source
            loss_tri /= self.n_source
            loss = loss_ce + loss_tri

            optimizer.zero_grad()
            loss.backward()
            # self.printGrad()
            optimizer.step()
            precs = 0.
            for j in range(self.n_source):
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
            self.writer.add_scalar('lr', lr, self.iter)

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

    def train2(self, epoch, data_loader_sources, optimizer, print_freq=50, train_iters=400):

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
            lr = self.get_lr(optimizer)
            print_now = False  # (i + 1) % (4 * print_freq) == 0

            # load data
            n_source_inputs = []
            for data_loader_source in data_loader_sources:
                source_inputs = data_loader_source.next()

                n_source_inputs.append(source_inputs)
            # target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            # arrange batch for domain-specific BN
            n_inputs = []
            n_targets = []
            for source_inputs in n_source_inputs:
                inputs, targets, _ = self._parse_data(source_inputs)
                n_inputs.append(inputs)
                n_targets.append(targets)

            inputs = torch.cat(tuple(n_inputs), dim=0)  # .contiguous()

            # forward
            probs, feats = self._forward(inputs, n_targets)

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

            if print_now:
                print('-' * 25)

            loss_ce /= self.n_source
            loss_tri /= self.n_source
            loss = loss_ce + loss_tri

            optimizer.zero_grad()
            loss.backward()
            # self.printGrad()
            optimizer.step()
            precs = 0.
            for j in range(self.n_source):
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
            self.writer.add_scalar('lr', lr, self.iter)

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


class MDE_MC_Trainer(MDE_Trainer):
    """
    MDE DG Trainer, data_loader_source has multiple data loader, with memory classifier
    """
    def __init__(self, args, model, n_num_classes, memories):
        self.memories = memories
        super(MDE_MC_Trainer, self).__init__(args, model, n_num_classes)

    def _forward(self, inputs, targets):
        bn_x, x, norm_feats = self.model(inputs)
        bn_x = torch.chunk(bn_x, self.n_source, dim=0)
        probs = []
        if self.classifiers:
            for i in range(self.n_source):
                prob = self.classifiers[i](bn_x[i], targets[i])
                probs.append(prob)
        return probs, x, norm_feats

    def get_WM_difference(self, loss=False):
        loss_diff = 0
        for i, mem in enumerate(self.memories):
            cls = self.classifiers[i]
            dist = self.calculate_distance(cls.weight, mem.module.features, loss=loss)
            self.writer.add_scalar('diff_{}'.format(i), dist.data, self.iter)
            # if self.iter % 50 == 0:
            #     print(dist)
            loss_diff += dist
        return loss_diff

    def calculate_distance(self, weight, features, loss=False):
        """ weight from classifier, features from memory cls"""
        if not loss:
            weight = weight.detach()
            features = features.detach()
        # mat_dist = euclidean_dist(F.normalize(weight), F.normalize(features))
        dist_diag = vector_cosine(weight, features)
        dist = dist_diag
        # dist = softmax_dist(weight, features)
        return dist

    def memory_record(self, epoch, save_freq=10):
        if epoch == 0:
            self.memory_dict = {}
            for i, mem in enumerate(self.memories):
                self.memory_dict[i] = [mem.module.features.cpu().data]
        else:
            for i, mem in enumerate(self.memories):
                self.memory_dict[i].append(mem.module.features.cpu().data)
            if (epoch + 1) % save_freq == 0:
                out_dir = osp.join(self.args.logs_dir, 'memory.pt')
                torch.save(self.memory_dict, out_dir)
                # torch.save(self.weight_dict, out_dir)
                print("save weight.pt at epoch {}".format(epoch))

    def train(self, epoch, data_loader_sources, optimizer, print_freq=50, train_iters=400):

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        losses_mem = AverageMeter()
        precisions = AverageMeter()
        losses_diff = AverageMeter()

        end = time.time()

        # self.memory_record(epoch)

        for i in range(train_iters):

            self.iter = epoch * train_iters + i
            self.writer.add_scalar('epoch', epoch, self.iter)


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

            inputs = torch.cat(tuple(n_inputs), dim=0).contiguous()

            # forward
            probs, feats, norm_feats = self._forward(inputs, n_targets)

            n_feats = torch.chunk(feats, self.n_source, dim=0)
            n_norm_feats = torch.chunk(norm_feats, self.n_source, dim=0)

            # classification+triplet
            loss_ce = torch.tensor(0., device='cuda')
            loss_tri = 0.
            loss_mem = 0.
            for j in range(self.n_source):
                if self.classifiers:
                    # loss_ce += 0.
                    loss_ce += self.criterion_ce(probs[j], n_targets[j])
                loss_tri += self.criterion_tri(n_feats[j], n_targets[j])
                loss_mem += self.memories[j](n_norm_feats[j], n_targets[j], self.iter).mean()

            loss_ce /= self.n_source
            loss_tri /= self.n_source
            loss_mem /= self.n_source

            # loss = loss_tri + loss_mem
            loss = loss_ce + loss_tri + loss_mem

            if self.classifiers:
                loss_diff = self.get_WM_difference(loss=self.args.loss_diff)
            else:
                loss_diff = torch.tensor(0.)
            loss_diff /= self.n_source
            # loss_diff *= 100
            if self.args.loss_diff:
                loss += loss_diff

            optimizer.zero_grad()
            loss.backward()
            # self.printGrad()
            optimizer.step()

            # update memory classifiers
            n_pid = self.args.batch_size // self.args.num_instances
            with torch.no_grad():
                for j, imgs, pids in zip(range(self.n_source), n_inputs, n_targets):
                    # imgs, pids, _ = self._parse_data(n_source_inputs[j])
                    _, _, f_new = self.model(imgs)  # bt * dim
                    # f_new_pid = torch.stack(torch.chunk(f_new, n_pid, dim=0), dim=0)  # pid * intances * dim
                    # f_new_mean = f_new_pid.mean(dim=1)  # pid * dim
                    # pids_mean_idx = torch.arange(0, self.args.batch_size, self.args.num_instances)
                    # pids_mean = pids[pids_mean_idx]
                    f_new_mean, pids_mean = f_new, pids
                    self.memories[j].module.MomentumUpdate(f_new_mean, pids_mean,
                                                           mean_update=self.args.mean_update)

            precs = 0.
            if self.classifiers:
                for j in range(self.n_source):
                    prec, = accuracy(probs[j].view(-1, probs[j].size(-1)).data, n_targets[j].data)
                    precs += prec[0]
                precs /= self.n_source

            losses.update(loss.item())
            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            losses_mem.update(loss_mem.item())
            losses_diff.update(loss_diff.item())
            precisions.update(precs)

            # add tensorboard
            self.writer.add_scalar('loss', loss.item(), self.iter)
            self.writer.add_scalar('loss_ce', loss_ce.item(), self.iter)
            self.writer.add_scalar('loss_tri', loss_tri.item(), self.iter)
            self.writer.add_scalar('loss_mem', loss_mem.item(), self.iter)
            self.writer.add_scalar('loss_diff', loss_diff.item(), self.iter)
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
                      'Loss_mem {:.3f} ({:.3f}) '
                      'Loss_diff {:.5} ({:.5}) '
                      'Prec_s {:.2%} ({:.2%}) '
                      'lr {:.3e}'
                      'momentum {:.3e}'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tri.val, losses_tri.avg,
                              losses_mem.val, losses_mem.avg,
                              losses_diff.val, losses_diff.avg,
                              precisions.val, precisions.avg,
                              self.get_lr(optimizer),
                              self.memories[0].module.momentum,
                              ))


