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
from ..layers.adaNorm_layer import calc_mean_std, mean_variance_norm


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
        self.args = args
        self.with_smm_loss = args.smm_mix_layer != 'AdaIN'
        if self.with_smm_loss:
            self.criterion_mse = nn.MSELoss().cuda()
        self.with_memory = args.global_memory or args.local_memory
        if self.with_memory:
            self.memory = args.memory

    def train(self, epoch, data_loader_sources, optimizer, print_freq=50, train_iters=400):

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        losses_ce_local = AverageMeter()
        losses_tri_local = AverageMeter()
        precisions_local = AverageMeter()
        losses_mem_local = AverageMeter()

        losses_ce_global = AverageMeter()
        losses_tri_global = AverageMeter()
        precisions_global = AverageMeter()
        losses_mem_global = AverageMeter()

        losses_content = AverageMeter()
        losses_style = AverageMeter()

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
            relabel = self.args.relabel
            g_targets = [relabel(j, n_targets[j]) for j in range(self.n_source)]
            n_outputs = g_targets

            device_num = torch.cuda.device_count()
            B, C, H, W = n_inputs[0].size()

            def reshape(inputs):
                return inputs.view(device_num, -1, C, H, W)

            n_inputs_2 = []
            for j, inputs in enumerate(n_inputs):
                n_inputs_2.append(reshape(inputs))

            inputs = torch.cat(tuple(n_inputs_2), dim=1).view(-1, C, H, W)


            # forward

            if self.with_smm_loss:
                (probs, g_probs, n_feats, n_feats_norm), mix_mean, mix_std, style_mean, style_std = self.model(inputs)
            else:
                probs, g_probs, n_feats, n_feats_norm = self.model(inputs)

            # n_feats = torch.chunk(feats, self.n_source, dim=0)
            if self.args.smm_scale:
                # expand label to match [[x, x_mix, ...](gpu1), [x, x_mix, ...](gpu2), ...]
                n_targets = [targets.view(device_num, -1) for targets in n_targets]
                size = len(self.args.smm_scale)
                n_targets = [targets.repeat((1, size)) for targets in n_targets]
                n_targets = [targets.view(-1) for targets in n_targets]
                g_targets = [relabel(j, n_targets[j]) for j in range(self.n_source)]

            loss = torch.tensor(0., device='cuda')
            # domain-specific loss, local
            if self.args.local_classifiers:
                loss_ce = torch.tensor(0., device='cuda')

                precs = 0.
                for j in range(self.n_source):
                    loss_ce += self.criterion_ce(probs[j], n_targets[j])
                    prec, = accuracy(probs[j].view(-1, probs[j].size(-1)).data, n_targets[j].data)
                    precs += prec[0]
                precs /= self.n_source
                loss_ce /= self.n_source
                loss_ce *= self.args.local_lambda
                loss += loss_ce
                losses_ce_local.update(loss_ce.item())
                precisions_local.update(precs)

            if self.args.local_memory:
                loss_mem_local = torch.tensor(0., device='cuda')
                for j in range(self.n_source):
                    loss_mem_local += self.memory(n_feats_norm[j], g_targets[j], j, self.iter).mean()
                loss_mem_local /= self.n_source
                loss_mem_local *= self.args.local_lambda
                loss += loss_mem_local

                losses_mem_local.update(loss_mem_local.item())

            if self.args.local_classifiers or self.args.local_memory:
                loss_tri = torch.tensor(0., device='cuda')
                for j in range(self.n_source):
                    loss_tri += self.criterion_tri(n_feats[j], n_targets[j])
                loss_tri /= self.n_source
                loss_tri *= self.args.local_lambda
                loss += loss_tri
                losses_tri_local.update(loss_tri.item())

            # domain-agnostic loss, global
            if self.args.global_classifier:
                g_targets = torch.cat(tuple(g_targets), dim=0)
                g_probs = torch.cat(tuple(g_probs), dim=0)
                loss_ce_g = self.criterion_ce(g_probs, g_targets) * self.args.global_lambda

                prec_g, = accuracy(g_probs.view(-1, g_probs.size(-1)).data, g_targets.data)
                loss += loss_ce_g

                losses_ce_global.update(loss_ce_g.item())
                precisions_global.update(prec_g[0])

            if self.args.global_memory:
                g_feats_norm = torch.cat(tuple(n_feats_norm), dim=0)
                g_targets = torch.cat(tuple(g_targets), dim=0)
                loss_mem_global = self.memory(g_feats_norm, g_targets, -1, self.iter) * self.args.global_lambda
                loss += loss_mem_global

                losses_mem_global.update(loss_mem_global.item())

            if self.args.global_classifier or self.args.global_memory:
                g_feats = torch.cat(tuple(n_feats), dim=0)
                g_targets = torch.cat(tuple(g_targets), dim=0)
                loss_tri_g = self.criterion_tri(g_feats, g_targets) * self.args.global_lambda
                loss += loss_tri_g
                losses_tri_global.update(loss_tri_g.item())


            if self.with_smm_loss:
                loss_content = torch.tensor(0., device='cuda')
                loss_style = torch.tensor(0., device='cuda')
                if self.args.smm_lambda_style > 0:
                    loss_style += self.criterion_mse(mix_mean, style_mean) + \
                                  self.criterion_mse(style_std, style_std)
                    loss_style *= self.args.smm_lambda_style
                loss += loss_content + loss_style

                losses_content.update(loss_content.item())
                losses_style.update(loss_style.item())

            losses.update(loss.item())

            optimizer.zero_grad()
            loss.backward()
            # self.printGrad()
            optimizer.step()

            # update memory classifiers
            if self.with_memory:
                with torch.no_grad():
                    outputs = self.model(inputs, identity=True)
                    f_new = torch.cat(outputs[3], dim=0)
                    pid_new = torch.cat(n_outputs, dim=0)
                    self.memory.module.MomentumUpdate(f_new, pid_new)
                    # for j, source_inputs in enumerate(n_source_inputs):
                    #     inputs, targets, _ = self._parse_data(source_inputs)
                    #     outputs = self.model(inputs)
                    #     f_new = F.normalize(outputs[2])
                    #     self.memory.module.MomentumUpdate(f_new, relabel(j, targets))

            # # add tensorboard
            # self.writer.add_scalar('loss', loss.item(), self.iter)
            # self.writer.add_scalar('loss_ce', loss_ce.item(), self.iter)
            # self.writer.add_scalar('loss_tri', loss_tri.item(), self.iter)
            # self.writer.add_scalar('Prec', precs, self.iter)
            # self.writer.add_scalar('lr', lr, self.iter)

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:

                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f}) '
                      'Data {:.3f} ({:.3f}) '
                      'lr {:.3e} '
                      'Loss {:.3f} ({:.3f}) '
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              self.get_lr(optimizer),
                              losses.val, losses.avg,))
                print('\t\t\t'
                      'Loss_tri_local {:.3f} ({:.3f}) '
                      'Loss_tri_global {:.3f} ({:.3f}) '
                      .format(losses_tri_local.val, losses_tri_local.avg,
                              losses_tri_global.val, losses_tri_global.avg,
                              ))
                
                if self.args.local_classifiers:
                    print('\t\t\t'
                          'Loss_ce_local {:.3f} ({:.3f}) '
                          'Prec_local {:.2%} ({:.2%}) '
                          .format(losses_ce_local.val, losses_ce_local.avg,
                                  precisions_local.val, precisions_local.avg,
                                  ))

                if self.args.global_classifier:
                    print('\t\t\t'
                          'Loss_ce_global {:.3f} ({:.3f}) '
                          'Prec_global {:.2%} ({:.2%}) '
                          .format(losses_ce_global.val, losses_ce_global.avg,
                                  precisions_global.val, precisions_global.avg,
                                  ))

                if self.with_memory:
                    print('\t\t\t'
                          'Loss_mem_local {:.3f} ({:.3f}) '
                          'Loss_mem_global {:.3f} ({:.3f}) '
                          .format(losses_mem_local.val, losses_mem_local.avg,
                                  losses_mem_global.val, losses_mem_global.avg,
                                  ))


                if self.with_smm_loss:
                    print(
                        'Loss_content {:.3f} ({:.3f}) '
                        'Loss_style {:.3f} ({:.3f}) '
                            .format(losses_content.val, losses_content.avg,
                                    losses_style.val, losses_style.avg,
                                    ))
                print(' ')

    def train_backup(self, epoch, data_loader_sources, optimizer, print_freq=50, train_iters=400):
        # 使用 1 个
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        losses_ce_local = AverageMeter()
        losses_tri_local = AverageMeter()
        precisions_local = AverageMeter()
        losses_mem_local = AverageMeter()

        losses_ce_global = AverageMeter()
        losses_tri_global = AverageMeter()
        precisions_global = AverageMeter()
        losses_mem_global = AverageMeter()

        losses_content = AverageMeter()
        losses_style = AverageMeter()

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
            relabel = self.args.relabel
            g_targets = [relabel(j, n_targets[j]) for j in range(self.n_source)]
            n_outputs = g_targets

            device_num = torch.cuda.device_count()
            B, C, H, W = n_inputs[0].size()

            def reshape(inputs):
                return inputs.view(device_num, -1, C, H, W)

            n_inputs_2 = []
            for j, inputs in enumerate(n_inputs):
                n_inputs_2.append(reshape(inputs))

            inputs = torch.cat(tuple(n_inputs_2), dim=1).view(-1, C, H, W)

            # forward

            if self.with_smm_loss:
                (probs, g_probs, n_feats), mix_mean, mix_std, style_mean, style_std = self.model(inputs)
            else:
                probs, g_probs, n_feats = self.model(inputs)

            # n_feats = torch.chunk(feats, self.n_source, dim=0)
            if self.args.smm_scale:
                # expand label to match [[x, x_mix, ...](gpu1), [x, x_mix, ...](gpu2), ...]
                n_targets = [targets.view(device_num, -1) for targets in n_targets]
                size = len(self.args.smm_scale)
                n_targets = [targets.repeat((1, size)) for targets in n_targets]
                n_targets = [targets.view(-1) for targets in n_targets]
                g_targets = [relabel(j, n_targets[j]) for j in range(self.n_source)]

            loss = torch.tensor(0., device='cuda')
            # domain-specific loss, local
            if self.args.local_classifiers:
                loss_ce = torch.tensor(0., device='cuda')

                precs = 0.
                for j in range(self.n_source):
                    loss_ce += self.criterion_ce(probs[j], n_targets[j])
                    prec, = accuracy(probs[j].view(-1, probs[j].size(-1)).data, n_targets[j].data)
                    precs += prec[0]
                precs /= self.n_source
                loss_ce /= self.n_source
                loss_ce *= self.args.local_lambda
                loss += loss_ce
                losses_ce_local.update(loss_ce.item())
                precisions_local.update(precs)

            if self.args.local_memory:
                loss_mem_local = torch.tensor(0., device='cuda')
                for j in range(self.n_source):
                    loss_mem_local += self.memory(F.normalize(n_feats[j]), g_targets[j], j, self.iter).mean()
                loss_mem_local /= self.n_source
                loss_mem_local *= self.args.local_lambda
                loss += loss_mem_local

                losses_mem_local.update(loss_mem_local.item())

            if self.args.local_classifiers or self.args.local_memory:
                loss_tri = torch.tensor(0., device='cuda')
                for j in range(self.n_source):
                    loss_tri += self.criterion_tri(n_feats[j], n_targets[j])
                loss_tri /= self.n_source
                loss_tri *= self.args.local_lambda
                loss += loss_tri
                losses_tri_local.update(loss_tri.item())

            # domain-agnostic loss, global
            if self.args.global_classifier:
                g_targets = torch.cat(tuple(g_targets), dim=0)
                g_probs = torch.cat(tuple(g_probs), dim=0)
                loss_ce_g = self.criterion_ce(g_probs, g_targets) * self.args.global_lambda

                prec_g, = accuracy(g_probs.view(-1, g_probs.size(-1)).data, g_targets.data)
                loss += loss_ce_g

                losses_ce_global.update(loss_ce_g.item())
                precisions_global.update(prec_g[0])

            if self.args.global_memory:
                g_feats = torch.cat(tuple(n_feats), dim=0)
                g_targets = torch.cat(tuple(g_targets), dim=0)
                loss_mem_global = self.memory(F.normalize(g_feats), g_targets, -1, self.iter) * self.args.global_lambda
                loss += loss_mem_global

                losses_mem_global.update(loss_mem_global.item())

            if self.args.local_classifiers or self.args.local_memory:
                g_feats = torch.cat(tuple(n_feats), dim=0)
                g_targets = torch.cat(tuple(g_targets), dim=0)
                loss_tri_g = self.criterion_tri(g_feats, g_targets) * self.args.global_lambda
                loss += loss_tri_g
                losses_tri_global.update(loss_tri_g.item())

            if self.with_smm_loss:
                loss_content = torch.tensor(0., device='cuda')
                loss_style = torch.tensor(0., device='cuda')
                if self.args.smm_lambda_style > 0:
                    loss_style += self.criterion_mse(mix_mean, style_mean) + \
                                  self.criterion_mse(style_std, style_std)
                    loss_style *= self.args.smm_lambda_style
                loss += loss_content + loss_style

                losses_content.update(loss_content.item())
                losses_style.update(loss_style.item())

            losses.update(loss.item())

            optimizer.zero_grad()
            loss.backward()
            # self.printGrad()
            optimizer.step()

            # update memory classifiers
            if self.with_memory:
                with torch.no_grad():
                    outputs = self.model(inputs, identity=True)
                    f_new = torch.cat(outputs[2], dim=0)
                    f_new = F.normalize(f_new)
                    pid_new = torch.cat(n_outputs, dim=0)
                    self.memory.module.MomentumUpdate(f_new, pid_new)
                    # for j, source_inputs in enumerate(n_source_inputs):
                    #     inputs, targets, _ = self._parse_data(source_inputs)
                    #     outputs = self.model(inputs)
                    #     f_new = F.normalize(outputs[2])
                    #     self.memory.module.MomentumUpdate(f_new, relabel(j, targets))

            # # add tensorboard
            # self.writer.add_scalar('loss', loss.item(), self.iter)
            # self.writer.add_scalar('loss_ce', loss_ce.item(), self.iter)
            # self.writer.add_scalar('loss_tri', loss_tri.item(), self.iter)
            # self.writer.add_scalar('Prec', precs, self.iter)
            # self.writer.add_scalar('lr', lr, self.iter)

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:

                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f}) '
                      'Data {:.3f} ({:.3f}) '
                      'lr {:.3e} '
                      'Loss {:.3f} ({:.3f}) '
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              self.get_lr(optimizer),
                              losses.val, losses.avg, ))
                print('\t\t\t'
                      'Loss_tri_local {:.3f} ({:.3f}) '
                      'Loss_tri_global {:.3f} ({:.3f}) '
                      .format(losses_tri_local.val, losses_tri_local.avg,
                              losses_tri_global.val, losses_tri_global.avg,
                              ))

                if self.args.local_classifiers:
                    print('\t\t\t'
                          'Loss_ce_local {:.3f} ({:.3f}) '
                          'Prec_local {:.2%} ({:.2%}) '
                          .format(losses_ce_local.val, losses_ce_local.avg,
                                  precisions_local.val, precisions_local.avg,
                                  ))

                if self.args.global_classifier:
                    print('\t\t\t'
                          'Loss_ce_global {:.3f} ({:.3f}) '
                          'Prec_global {:.2%} ({:.2%}) '
                          .format(losses_ce_global.val, losses_ce_global.avg,
                                  precisions_global.val, precisions_global.avg,
                                  ))

                if self.with_memory:
                    print('\t\t\t'
                          'Loss_mem_local {:.3f} ({:.3f}) '
                          'Loss_mem_global {:.3f} ({:.3f}) '
                          .format(losses_mem_local.val, losses_mem_local.avg,
                                  losses_mem_global.val, losses_mem_global.avg,
                                  ))

                if self.with_smm_loss:
                    print(
                        'Loss_content {:.3f} ({:.3f}) '
                        'Loss_style {:.3f} ({:.3f}) '
                            .format(losses_content.val, losses_content.avg,
                                    losses_style.val, losses_style.avg,
                                    ))
                print(' ')


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

    def train(self, epoch, data_loader_sources, optimizer, print_freq=50, train_iters=400):
        # 把 batch 与机器对应上
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        losses_mem = AverageMeter()
        precisions = AverageMeter()

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

            data_time.update(time.time() - end)

            # process inputs
            n_inputs = []
            n_targets = []
            for j, source_inputs in enumerate(n_source_inputs):
                inputs, targets, _ = self._parse_data(source_inputs)
                n_inputs.append(inputs)
                n_targets.append(targets)


            # arrange batch for domain-specific operations
            # such as DSBN, Memory(DataParallel)
            # e.g. data 顺序 [0,0,1,1,2,2] -> [0,1,2,0,1,2],
            # Then gpu0: [0,1,2] and gpu1: [0,1,2]
            # else gpu0: [0,0,1] and gpu1: [1,2,2]
            device_num = torch.cuda.device_count()
            B, C, H, W = n_inputs[0].size()

            def reshape(inputs):
                return inputs.view(device_num, -1, C, H, W)

            n_inputs_2 = []
            for j, inputs in enumerate(n_inputs):
                n_inputs_2.append(reshape(inputs))

            inputs = torch.cat(tuple(n_inputs_2), dim=1).view(-1, C, H, W)

            # forward
            n_probs, _, n_feats, n_norm_feats = self.model(inputs)


            # classification+triplet
            loss_ce = torch.tensor(0., device='cuda')
            loss_tri = 0.
            loss_mem = 0.
            for j in range(self.n_source):
                if self.classifiers:
                    # loss_ce += 0.
                    loss_ce += self.criterion_ce(n_probs[j], n_targets[j])
                loss_tri += self.criterion_tri(n_feats[j], n_targets[j])
                loss_mem += self.memories[j](n_norm_feats[j], n_targets[j], self.iter).mean()

            loss_ce /= self.n_source
            loss_tri /= self.n_source
            loss_mem /= self.n_source

            # loss = loss_tri + loss_mem
            loss = loss_ce + loss_tri + loss_mem

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update memory classifiers
            with torch.no_grad():
                _, _, _, n_norm_feats = self.model(inputs)
                for j, feat_new, pids in zip(range(self.n_source), n_norm_feats, n_targets):
                    self.memories[j].MomentumUpdate(feat_new, pids,
                                                    mean_update=self.args.mean_update)

            precs = 0.
            if self.classifiers:
                for j in range(self.n_source):
                    prec, = accuracy(n_probs[j].view(-1, n_probs[j].size(-1)).data, n_targets[j].data)
                    precs += prec[0]
                precs /= self.n_source

            losses.update(loss.item())
            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            losses_mem.update(loss_mem.item())
            precisions.update(precs)

            # add tensorboard
            self.writer.add_scalar('loss', loss.item(), self.iter)
            self.writer.add_scalar('loss_ce', loss_ce.item(), self.iter)
            self.writer.add_scalar('loss_tri', loss_tri.item(), self.iter)
            self.writer.add_scalar('loss_mem', loss_mem.item(), self.iter)
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
                      'Prec_s {:.2%} ({:.2%}) '
                      'lr {:.3e}'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tri.val, losses_tri.avg,
                              losses_mem.val, losses_mem.avg,
                              precisions.val, precisions.avg,
                              self.get_lr(optimizer),
                              ))

    def train_back(self, epoch, data_loader_sources, optimizer, print_freq=50, train_iters=400):
        # 把 batch 与机器对应上，此版本需要把 feature 恢复，而最新方法在 model 中进行了 chunk，
        # 从而对其 domian 之间的 features
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        losses_mem = AverageMeter()
        precisions = AverageMeter()

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

            data_time.update(time.time() - end)

            # process inputs
            n_inputs = []
            n_targets = []
            # test = []
            for j, source_inputs in enumerate(n_source_inputs):
                inputs, targets, _ = self._parse_data(source_inputs)
                n_inputs.append(inputs)
                n_targets.append(targets)
                # test.append(torch.zeros_like(targets).to(targets.device) + j)

            # arrange batch for domain-specific operations
            # such as DSBN, Memory(DataParallel)
            # e.g. data 顺序 [0,0,1,1,2,2] -> [0,1,2,0,1,2],
            # Then gpu0: [0,1,2] and gpu1: [0,1,2]
            # else gpu0: [0,0,1] and gpu1: [1,2,2]
            device_num = torch.cuda.device_count()
            B, C, H, W = n_inputs[0].size()

            def reshape(inputs):
                return inputs.view(device_num, -1, C, H, W)

            n_inputs_2 = []
            # n_targets_2 = []  # reshape 后的 inputs
            for j, inputs in enumerate(n_inputs):
                n_inputs_2.append(reshape(inputs))
                # n_targets_2.append(test[j].view(device_num, -1))

            #### 目前不是什么原因，loss一直不降，代码哪存在问题
            inputs = torch.cat(tuple(n_inputs_2), dim=1).view(-1, C, H, W)
            # tmp_targets = torch.cat(tuple(n_targets_2), dim=1).view(-1)

            # tst
            # t_inputs = torch.cat(tuple(n_inputs), dim=0)
            # _, t_x, _ = self.model(t_inputs)
            # inputs_ = inputs.view(12, -1).sum(dim=1)
            # t_inputs_ = t_inputs.view(12, -1).sum(dim=1)

            # forward
            bn_x, x, norm_feats = self.model(inputs)
            x.retain_grad()

            def recover(xs):
                # 把 [0,1,2,0,1,2] 的 data 顺序恢复为 [0,0,1,1,2,2]
                B, C = xs.size()
                xs = xs.view(device_num, -1, C)
                xs = torch.chunk(xs, device_num, dim=0)
                t_xs = [per_x.view(-1, 1, C) for per_x in xs]
                # for per_x in xs:
                #     per_x = per_x.view(-1, 1, C)
                #     t_xs.append(per_x)
                xs = torch.cat(tuple(t_xs), dim=1).view(-1, C)
                xs = torch.chunk(xs, self.n_source, dim=0)
                return xs
            bn_x, n_feats, n_norm_feats = recover(bn_x), recover(x), recover(norm_feats)

            probs = []
            if self.classifiers:
                for j in range(self.n_source):
                    prob = self.classifiers[j](bn_x[j], n_targets[j])
                    probs.append(prob)

            # classification+triplet
            loss_ce = torch.tensor(0., device='cuda')
            loss_tri = 0.
            loss_mem = 0.
            for j in range(self.n_source):
                if self.classifiers:
                    # loss_ce += 0.
                    loss_ce += self.criterion_ce(probs[j], n_targets[j])
                loss_tri += self.criterion_tri(n_feats[j], n_targets[j])
                loss_mem += self.memories[j](n_norm_feats[j], n_targets[j]).mean()

            loss_ce /= self.n_source
            loss_tri /= self.n_source
            loss_mem /= self.n_source

            # loss = loss_tri + loss_mem
            loss = loss_ce + loss_tri + loss_mem

            optimizer.zero_grad()
            loss.backward()
            # self.printGrad()
            optimizer.step()

            # import IPython
            # IPython.embed()
            # print(x)


            # update memory classifiers
            n_pid = self.args.batch_size // self.args.num_instances
            with torch.no_grad():
                _, _, norm_feats = self.model(inputs)
                n_norm_feats = recover(norm_feats)
                for j, feat_new, pids in zip(range(self.n_source), n_norm_feats, n_targets):
                    self.memories[j].MomentumUpdate(feat_new, pids,
                                                    mean_update=self.args.mean_update)
                # for j, source_inputs in enumerate(n_source_inputs):
                #     imgs, pids, _ = self._parse_data(source_inputs)
                # for j, imgs, pids in zip(range(self.n_source), n_inputs_2, n_targets):
                #     # imgs, pids, _ = self._parse_data(n_source_inputs[j])
                #     _, _, f_new = self.model(imgs)  # bt * dim
                #     # f_new_pid = torch.stack(torch.chunk(f_new, n_pid, dim=0), dim=0)  # pid * intances * dim
                #     # f_new_mean = f_new_pid.mean(dim=1)  # pid * dim
                #     # pids_mean_idx = torch.arange(0, self.args.batch_size, self.args.num_instances)
                #     # pids_mean = pids[pids_mean_idx]
                #     f_new_mean, pids_mean = f_new, pids
                #     self.memories[j].MomentumUpdate(f_new_mean, pids_mean,
                #                                            mean_update=self.args.mean_update)

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
            precisions.update(precs)

            # add tensorboard
            self.writer.add_scalar('loss', loss.item(), self.iter)
            self.writer.add_scalar('loss_ce', loss_ce.item(), self.iter)
            self.writer.add_scalar('loss_tri', loss_tri.item(), self.iter)
            self.writer.add_scalar('loss_mem', loss_mem.item(), self.iter)
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
                      'Prec_s {:.2%} ({:.2%}) '
                      'lr {:.3e}'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tri.val, losses_tri.avg,
                              losses_mem.val, losses_mem.avg,
                              precisions.val, precisions.avg,
                              self.get_lr(optimizer),
                              ))
