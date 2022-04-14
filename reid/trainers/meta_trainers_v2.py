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
import numpy as np
import copy


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
        self.meta_beta = args.meta_beta
        self.with_memory = args.global_memory or args.local_memory
        if self.with_memory:
            self.memory = args.memory

    def train(self, epoch, data_loader_sources, optimizer, print_freq=50, train_iters=400):

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        inner_losses = AverageMeter()
        outer_losses = AverageMeter()

        # local
        inner_losses_ce_local = AverageMeter()
        outer_losses_ce_local = AverageMeter()
        inner_losses_tri_local = AverageMeter()
        outer_losses_tri_local = AverageMeter()
        inner_precisions_local = AverageMeter()
        outer_precisions_local = AverageMeter()

        inner_losses_mem_local = AverageMeter()
        outer_losses_mem_local = AverageMeter()

        # global
        inner_losses_ce_global = AverageMeter()
        outer_losses_ce_global = AverageMeter()
        inner_losses_tri_global = AverageMeter()
        outer_losses_tri_global = AverageMeter()
        inner_precisions_global = AverageMeter()
        outer_precisions_global = AverageMeter()

        inner_losses_mem_global = AverageMeter()
        outer_losses_mem_global = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            self.iter = epoch * train_iters + i
            self.writer.add_scalar('epoch', epoch, self.iter)
            lr = self.get_lr(optimizer)

            # load data
            n_source_inputs = [loader.next() for loader in data_loader_sources]
            # for data_loader_source in data_loader_sources:
            #     source_inputs = data_loader_source.next()
            #     n_source_inputs.append(source_inputs)
            # target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            n_inputs = []
            n_targets = []
            for source_inputs in n_source_inputs:
                inputs, targets, _ = self._parse_data(source_inputs)
                n_inputs.append(inputs)
                n_targets.append(targets)

            device_num = torch.cuda.device_count()
            B, C, H, W = n_inputs[0].size()

            def reshape(inputs):
                return inputs.view(device_num, -1, C, H, W)

            # select test domain
            test_domain = np.random.choice(range(self.n_source))
            train_domains = [d for d in range(self.n_source) if d != test_domain]

            objective = 0.
            optimizer.zero_grad()
            for p in self.model.parameters():
                if p.grad is None:
                    p.grad = torch.zeros_like(p)

            """
            inner update phase
            """
            inner_net = copy.deepcopy(self.model)
            inner_optimizer = torch.optim.Adam(inner_net.module.get_params(), lr=lr, weight_decay=5e-4)

            # train data processing
            train_inputs = [n_inputs[domain_id] for domain_id in train_domains]
            train_targets = [n_targets[domain_id] for domain_id in train_domains]
            relabel = self.args.relabel
            g_train_targets = [relabel(domain_id, targets) for (domain_id, targets) in zip(train_domains, train_targets)]
            train_inputs_2 = []
            for inputs in train_inputs:
                train_inputs_2.append(reshape(inputs))
            inputs = torch.cat(tuple(train_inputs_2), dim=1).view(-1, C, H, W)

            # forward
            probs, g_probs, n_feats, n_feats_norm = inner_net(inputs, meta_train=True, meta_test_domain=test_domain)

            # expand label for matching SMM output
            if self.args.smm_scale:
                train_targets = [targets.view(device_num, -1) for targets in train_targets]
                # size = len(self.args.smm_scale)
                size = self.args.scale_len
                train_targets = [targets.repeat((1, size)) for targets in train_targets]
                train_targets = [targets.view(-1) for targets in train_targets]
                g_train_targets = [relabel(domain_id, targets) for (domain_id, targets) in zip(train_domains, train_targets)]

            inner_loss = torch.tensor(0., device='cuda')
            # domain-specific train loss, local
            if self.args.local_classifiers:
                inner_loss_ce = torch.tensor(0., device='cuda')
                inner_prec = 0.
                for j in range(self.n_source - 1):
                    inner_loss_ce += self.criterion_ce(probs[j], train_targets[j])
                    prec, = accuracy(probs[j].view(-1, probs[j].size(-1)).data, train_targets[j].data)
                    inner_prec += prec[0]
                inner_prec /= (self.n_source - 1)
                inner_loss_ce /= (self.n_source - 1)
                inner_loss_ce *= self.args.local_lambda
                inner_loss += inner_loss_ce

                inner_losses_ce_local.update(inner_loss_ce.item())
                inner_precisions_local.update(inner_prec)

            if self.args.local_memory:
                inner_loss_mem_local = torch.tensor(0., device='cuda')
                for j, train_domain in enumerate(train_domains):
                    inner_loss_mem_local += self.memory(n_feats_norm[j], g_train_targets[j], train_domain, self.iter).mean()
                inner_loss_mem_local /= (self.n_source - 1)
                inner_loss_mem_local *= self.args.local_lambda
                inner_loss += inner_loss_mem_local

                inner_losses_mem_local.update(inner_loss_mem_local.item())

            if self.args.local_classifiers or self.args.local_memory:
                inner_loss_tri = torch.tensor(0., device='cuda')
                for j in range(self.n_source - 1):
                    inner_loss_tri += self.criterion_tri(n_feats[j], train_targets[j])
                inner_loss_tri /= (self.n_source - 1)
                inner_loss_tri *= self.args.local_lambda
                inner_loss += inner_loss_tri
                inner_losses_tri_local.update(inner_loss_tri.item())

            # domain-agnostic loss, global
            if self.args.global_classifier:
                g_targets = torch.cat(tuple(g_train_targets), dim=0)
                g_probs = torch.cat(tuple(g_probs), dim=0)
                inner_loss_ce_g = self.criterion_ce(g_probs, g_targets) * self.args.global_lambda
                prec_g, = accuracy(g_probs.view(-1, g_probs.size(-1)).data, g_targets.data)
                inner_loss += inner_loss_ce_g

                inner_losses_ce_global.update(inner_loss_ce_g.item())
                inner_precisions_global.update(prec_g[0])

            if self.args.global_memory:
                g_feats_norm = torch.cat(tuple(n_feats_norm), dim=0)
                g_targets = torch.cat(tuple(g_train_targets), dim=0)
                inner_loss_mem_global = self.memory(g_feats_norm, g_targets, -1, self.iter).mean()
                inner_loss_mem_global *= self.args.global_lambda
                inner_loss += inner_loss_mem_global

                inner_losses_mem_global.update(inner_loss_mem_global.item())

            if self.args.global_classifier or self.args.global_memory:
                g_feats = torch.cat(tuple(n_feats), dim=0)
                g_targets = torch.cat(tuple(g_train_targets), dim=0)
                inner_loss_tri_g = self.criterion_tri(g_feats, g_targets) * self.args.global_lambda
                inner_loss += inner_loss_tri_g
                inner_losses_tri_global.update(inner_loss_tri_g.item())

            inner_losses.update(inner_loss.item())
            
            inner_optimizer.zero_grad()
            inner_loss.backward()
            inner_optimizer.step()

            # Now inner_net has accumulated gradients Gi
            # The clone-network (inner_net) has now parameters P - lr * Gi
            # Adding the gradient Gi to original network
            for p_outer, p_inner in zip(self.model.module.get_params(), inner_net.module.get_params()):
                if p_inner.grad is not None:
                    assert p_inner.grad.data.shape == p_outer.grad.data.shape
                    p_outer.grad.data.add_(p_inner.grad.data / self.n_source)

            objective += inner_loss.item()

            """
            outer update phase
            """
            test_inputs = n_inputs[test_domain]
            test_targets = n_targets[test_domain]
            g_targets = relabel(test_domain, test_targets)

            probs, g_probs, feats, feats_norm = inner_net(test_inputs, meta_train=False, meta_test_domain=test_domain)

            outer_loss = torch.tensor(0., device="cuda")
            if self.args.local_classifiers:
                outer_loss_ce = self.criterion_ce(probs, test_targets) * self.args.local_lambda
                prec, = accuracy(probs.view(-1, probs.size(-1)).data, test_targets.data)
                outer_loss += outer_loss_ce

                outer_losses_ce_local.update(outer_loss_ce.item())
                outer_precisions_local.update(prec[0])

            if self.args.local_memory:
                outer_loss_mem_local = self.memory(feats_norm, g_targets, test_domain, self.iter).mean()
                outer_loss_mem_local *= self.args.local_lambda
                outer_loss += outer_loss_mem_local

                outer_losses_mem_local.update(outer_loss_mem_local.item())

            if self.args.local_classifiers or self.args.local_memory:
                outer_loss_tri = self.criterion_tri(feats, test_targets) * self.args.local_lambda
                outer_loss += outer_loss_tri
                outer_losses_tri_local.update(outer_loss_tri.item())

            if self.args.global_classifier:
                # relabel = self.args.relabel
                # g_targets = relabel(test_domain, n_targets[test_domain])
                outer_loss_ce_g = self.criterion_ce(g_probs, g_targets) * self.args.global_lambda
                prec_g, = accuracy(g_probs.view(-1, g_probs.size(-1)).data, g_targets.data)
                outer_loss += outer_loss_ce_g

                outer_losses_ce_global.update(outer_loss_ce_g.item())
                outer_precisions_global.update(prec_g[0])

            if self.args.global_memory:
                outer_loss_mem_global = self.memory(feats_norm, g_targets, -1, self.iter).mean()
                outer_loss_mem_global *= self.args.global_lambda
                outer_loss += outer_loss_mem_global

                outer_losses_mem_global.update(outer_loss_mem_global.item())

            if self.args.global_classifier or self.args.global_memory:
                outer_loss_tri_g = self.criterion_tri(feats, g_targets) * self.args.global_lambda
                outer_loss += outer_loss_tri_g
                outer_losses_tri_global.update(outer_loss_tri_g.item())

            outer_losses.update(outer_loss.item())

            objective += self.meta_beta * outer_loss.item()
            losses.update(objective)

            grad_outer = torch.autograd.grad(outer_loss, inner_net.module.get_params(), allow_unused=True)

            for p_outer, g_outer in zip(self.model.module.get_params(), grad_outer):
                if g_outer is not None:
                    assert g_outer.data.shape == p_outer.grad.data.shape
                    p_outer.grad.data.add_(self.meta_beta * g_outer.data / self.n_source)

            # def copy_statics(src_model, dst_model):
            #     for (n_outer, m_outer), (n_inner, m_inner) in zip(dst_model.named_children(), src_model.named_children()):
            #         assert n_outer == n_inner
            #         if isinstance(m_outer, nn.BatchNorm2d) or isinstance(m_outer, nn.BatchNorm1d):
            #             m_outer.running_mean.data = m_inner.running_mean.data.clone()
            #             m_outer.running_var.data = m_inner.running_var.data.clone()
            #             m_outer.num_batches_tracked.data = m_inner.num_batches_tracked.data.clone()
            #         else:
            #             copy_statics(m_inner, m_outer)
            #
            # copy_statics(inner_net, self.model)

            for (n_outer, m_outer), (n_inner, m_inner) in zip(self.model.named_modules(), inner_net.named_modules()):
                assert n_outer == n_inner
                if isinstance(m_outer, nn.BatchNorm2d):
                    m_outer.running_mean.data = m_inner.running_mean.data.clone()
                    m_outer.running_var.data = m_inner.running_var.data.clone()
                    m_outer.num_batches_tracked.data = m_inner.num_batches_tracked.data.clone()
                if isinstance(m_outer, nn.BatchNorm1d):
                    m_outer.running_mean.data = m_inner.running_mean.data.clone()
                    m_outer.running_var.data = m_inner.running_var.data.clone()
                    m_outer.num_batches_tracked.data = m_inner.num_batches_tracked.data.clone()

            optimizer.step()

            # update memory classifiers
            if self.with_memory:
                with torch.no_grad():
                    for (j, inputs), targets in zip(enumerate(n_inputs), n_targets):
                        g_targets = relabel(j, targets)
                        _, _, _, f_new = self.model(inputs, meta_train=False, meta_test_domain=j)
                        self.memory.module.MomentumUpdate(f_new, g_targets)

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
                      'lr {:.3e} \n'
                      '\t\t\t'
                      'Loss {:.3f} ({:.3f}) '
                      'Inner_Loss {:.3f} ({:.3f}) '
                      'Outer_Loss {:.3f} ({:.3f}) '
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              lr,
                              losses.val, losses.avg,
                              inner_losses.val, inner_losses.avg,
                              outer_losses.val, outer_losses.avg,
                              ))

                if self.args.local_classifiers or self.args.local_memory:
                    print('\t\t\t'
                          'Inner_Loss_tri_local {:.3f} ({:.3f}) '
                          'Outer_Loss_tri_local {:.3f} ({:.3f}) '
                          .format(inner_losses_tri_local.val, inner_losses_tri_local.avg,
                                  outer_losses_tri_local.val, outer_losses_tri_local.avg,
                                  ))

                if self.args.local_classifiers:
                    print('\t\t\t'
                          'Inner_Loss_ce_local {:.3f} ({:.3f}) '
                          'Inner_Prec_local {:.2%} ({:.2%}) \n'
                          '\t\t\t'
                          'Outer_Loss_ce_local {:.3f} ({:.3f}) '
                          'Outer_Prec_local {:.2%} ({:.2%}) '
                          .format(inner_losses_ce_local.val, inner_losses_ce_local.avg,
                                  inner_precisions_local.val, inner_precisions_local.avg,
                                  outer_losses_ce_local.val, outer_losses_ce_local.avg,
                                  outer_precisions_local.val, outer_precisions_local.avg,
                                  ))

                if self.args.local_memory:
                    print('\t\t\t'
                          'Inner_Loss_mem_local {:.3f} ({:.3f}) '
                          'Outer_Loss_mem_local {:.3f} ({:.3f}) '
                          .format(inner_losses_mem_local.val, inner_losses_mem_local.avg,
                                  outer_losses_mem_local.val, outer_losses_mem_local.avg,
                                  ))

                if self.args.global_classifier or self.args.global_memory:
                    print('\t\t\t'
                          'Inner_Loss_tri_global {:.3f} ({:.3f}) '
                          'Outer_Loss_tri_global {:.3f} ({:.3f}) '
                          .format(inner_losses_tri_global.val, inner_losses_tri_global.avg,
                                  outer_losses_tri_global.val, outer_losses_tri_global.avg,
                                  ))

                if self.args.global_classifier:
                    print('\t\t\t'
                          'Inner_Loss_ce_global {:.3f} ({:.3f}) '                         
                          'Inner_Prec_global {:.2%} ({:.2%}) \n'
                          '\t\t\t'
                          'Outer_Loss_ce_global {:.3f} ({:.3f}) '                       
                          'Outer_Prec_global {:.2%} ({:.2%}) '
                          .format(inner_losses_ce_global.val, inner_losses_ce_global.avg,
                                  inner_precisions_global.val, inner_precisions_global.avg,
                                  outer_losses_ce_global.val, outer_losses_ce_global.avg,
                                  outer_precisions_global.val, outer_precisions_global.avg,
                                  ))

                if self.args.global_memory:
                    print('\t\t\t'
                          'Inner_Loss_mem_global {:.3f} ({:.3f}) '
                          'Outer_Loss_mem_global {:.3f} ({:.3f}) '
                          .format(inner_losses_mem_global.val, inner_losses_mem_global.avg,
                                  outer_losses_mem_global.val, outer_losses_mem_global.avg,
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
            n_probs, n_feats, n_norm_feats = self.model(inputs)


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
                _, _, n_norm_feats = self.model(inputs)
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
