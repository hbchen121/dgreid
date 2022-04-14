from __future__ import print_function, absolute_import
import time
import torch
from ..utils.meters import AverageMeter
from ..evaluation_metrics import accuracy
from ..loss import TripletLoss, CrossEntropyLabelSmooth, TripletLossXBM, DivLoss, BridgeFeatLoss, BridgeProbLoss
from reid.solver.utils import get_lr as get_lr_from_optimizer

class TrainerBase(object):
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.writer = args.writer
        self.iter = 0
        # self.logs = self.build_logs()

    def train(self, epoch, data_loader_source, optimizer, print_freq=50, train_iters=400):
        pass

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.model(inputs)

    def get_lr(self, optimizer):
        return get_lr_from_optimizer(optimizer)

    # def build_logs(self):
    #     batch_time = AverageMeter()
    #     data_time = AverageMeter()
    #     losses = AverageMeter()
    #     losses_ce = AverageMeter()
    #     losses_tri = AverageMeter()
    #     precisions = AverageMeter()
    #     logs = {
    #         'Time': batch_time,
    #         'Data': data_time,
    #         'Loss': losses,
    #         'Loss_ce': losses_ce,
    #         'Loss_tri': losses_tri,
    #         'Prec_s': precisions,
    #     }
    #     return logs
    #
    # def update_log(self, ):
    #     return



class Baseline_Trainer(TrainerBase):
    def __init__(self, args, model, num_classes):
        super(Baseline_Trainer, self).__init__(args, model)
        self.model = model
        self.num_classes = num_classes
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_tri = TripletLoss(margin=args.margin).cuda()

    def train(self, epoch, data_loader_source, optimizer, print_freq=50, train_iters=400):

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            source_inputs = data_loader_source.next()
            # target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, targets, _ = self._parse_data(source_inputs)

            # forward
            prob, feats = self._forward(inputs)

            # classification+triplet
            loss_ce = self.criterion_ce(prob, targets)
            loss_tri = self.criterion_tri(feats, targets)

            loss = loss_ce + loss_tri

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prec, = accuracy(prob.view(-1, prob.size(-1)).data, targets.data)

            losses.update(loss.item())
            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            precisions.update(prec[0])

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


class Memory_Trainer(Baseline_Trainer):
    def __init__(self, args, model, num_classes, memory):
        super(Memory_Trainer, self).__init__(args, model, num_classes)
        self.memory = memory

    def train(self, epoch, data_loader_source, optimizer, print_freq=50, train_iters=400):
        # self.criterion_ce = CrossEntropyLabelSmooth(source_classes).cuda()

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        losses_mem = AverageMeter()
        precisions = AverageMeter()
        # precisions_t = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            source_inputs = data_loader_source.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, targets, _ = self._parse_data(source_inputs)

            # forward
            prob, feats, norm_feats = self._forward(inputs)

            # classification + triplet + memory
            if self.args.classifier:
                loss_ce = self.criterion_ce(prob, targets)
            else:
                loss_ce = torch.tensor(0.).to(feats.device)
            loss_tri = self.criterion_tri(feats, targets)
            loss_mem = self.memory(norm_feats, targets).mean()

            loss = loss_ce + loss_tri + loss_mem

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update memory classifier
            with torch.no_grad():
                imgs, pids, _ = self._parse_data(source_inputs)
                _, _, f_new = self.model(imgs)
                self.memory.module.MomentumUpdate(f_new, pids)

            if self.args.classifier:
                prec, = accuracy(prob.view(-1, prob.size(-1)).data, targets.data)
            else:
                prec = [0.]

            losses.update(loss.item())
            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            losses_mem.update(loss_mem.item())
            precisions.update(prec[0])

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



class UDA_Baseline_Trainer(object):
    """
    UDA baseline
    """
    def __init__(self, args, model, xbm, num_classes, margin=None):
        super(UDA_Baseline_Trainer, self).__init__()
        self.model = model
        self.xbm = xbm
        self.args = args
        self.num_classes = num_classes
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_tri = TripletLoss(margin=args.margin).cuda()
        self.criterion_tri_xbm = TripletLossXBM(margin=args.margin)

    def train(self, epoch, data_loader_source, data_loader_target, source_classes, target_classes,
              optimizer, print_freq=50, train_iters=400, use_xbm=False):
        self.criterion_ce = CrossEntropyLabelSmooth(source_classes + target_classes).cuda()

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        losses_xbm = AverageMeter()
        precisions_s = AverageMeter()
        precisions_t = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            source_inputs = data_loader_source.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            s_inputs, s_targets, _ = self._parse_data(source_inputs)
            t_inputs, t_targets, t_indexes = self._parse_data(target_inputs)

            # arrange batch for domain-specific BN
            device_num = torch.cuda.device_count()
            B, C, H, W = s_inputs.size()

            def reshape(inputs):
                return inputs.view(device_num, -1, C, H, W)

            s_inputs, t_inputs = reshape(s_inputs), reshape(t_inputs)
            inputs = torch.cat((s_inputs, t_inputs), 1).view(-1, C, H, W)

            targets = torch.cat((s_targets.view(device_num, -1), t_targets.view(device_num, -1)), 1)
            targets = targets.view(-1)
            # forward
            prob, feats = self._forward(inputs) 
            prob = prob[:, 0:source_classes + target_classes]
        
            # split feats
            ori_feats = feats.view(device_num, -1, feats.size(-1))
            feats_s, feats_t = ori_feats.split(ori_feats.size(1) // 2, dim=1)
            ori_feats = torch.cat((feats_s, feats_t), 1).view(-1, ori_feats.size(-1))

            # classification+triplet
            loss_ce = self.criterion_ce(prob, targets)
            loss_tri = self.criterion_tri(ori_feats, targets)

            # enqueue and dequeue for xbm
            if use_xbm:
                self.xbm.enqueue_dequeue(ori_feats.detach(), targets.detach())
                xbm_feats, xbm_targets = self.xbm.get()
                loss_xbm = self.criterion_tri_xbm(ori_feats, targets, xbm_feats, xbm_targets)
                losses_xbm.update(loss_xbm.item())
                loss = loss_ce + loss_tri + loss_xbm 
            else:
                loss = loss_ce + loss_tri 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ori_prob = prob.view(device_num, -1, prob.size(-1))
            prob_s, prob_t = ori_prob.split(ori_prob.size(1) // 2, dim=1)
            prob_s, prob_t = prob_s.contiguous(), prob_t.contiguous()
            prec_s, = accuracy(prob_s.view(-1, prob_s.size(-1)).data, s_targets.data)
            prec_t, = accuracy(prob_t.view(-1, prob_s.size(-1)).data, t_targets.data)

            losses.update(loss.item())
            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            precisions_s.update(prec_s[0])
            precisions_t.update(prec_t[0])

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:

                if use_xbm:
                    print('Epoch: [{}][{}/{}]\t'
                          'Time {:.3f} ({:.3f}) '
                          'Data {:.3f} ({:.3f}) '
                          'Loss {:.3f} ({:.3f}) '
                          'Loss_ce {:.3f} ({:.3f}) '
                          'Loss_tri {:.3f} ({:.3f}) '
                          'Loss_xbm {:.3f} ({:.3f}) '
                          'Prec_s {:.2%} ({:.2%}) '
                          'Prec_t {:.2%} ({:.2%}) '
                          .format(epoch, i + 1, len(data_loader_target),
                                  batch_time.val, batch_time.avg,
                                  data_time.val, data_time.avg,
                                  losses.val, losses.avg,
                                  losses_ce.val, losses_ce.avg,
                                  losses_tri.val, losses_tri.avg,
                                  losses_xbm.val, losses_xbm.avg,
                                  precisions_s.val, precisions_s.avg,
                                  precisions_t.val, precisions_t.avg
                                  ))
                else:
                    print('Epoch: [{}][{}/{}]\t'
                          'Time {:.3f} ({:.3f}) '
                          'Data {:.3f} ({:.3f}) '
                          'Loss {:.3f} ({:.3f}) '
                          'Loss_ce {:.3f} ({:.3f}) '
                          'Loss_tri {:.3f} ({:.3f}) '
                          'Prec_s {:.2%} ({:.2%}) '
                          'Prec_t {:.2%} ({:.2%}) '
                          .format(epoch, i + 1, len(data_loader_target),
                                  batch_time.val, batch_time.avg,
                                  data_time.val, data_time.avg,
                                  losses.val, losses.avg,
                                  losses_ce.val, losses_ce.avg,
                                  losses_tri.val, losses_tri.avg,
                                  precisions_s.val, precisions_s.avg,
                                  precisions_t.val, precisions_t.avg
                                  ))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.model(inputs)


class IDM_Trainer(object):
    def __init__(self, model, xbm, num_classes, margin=None, mu1=1.0, mu2=1.0, mu3=1.0):
        super(IDM_Trainer, self).__init__()
        self.model = model
        self.xbm = xbm
        self.mu1 = mu1
        self.mu2 = mu2
        self.mu3 = mu3
        self.num_classes = num_classes
        self.criterion_ce = BridgeProbLoss(num_classes).cuda()
        self.criterion_tri = TripletLoss(margin=margin).cuda()
        self.criterion_tri_xbm = TripletLossXBM(margin=margin)
        self.criterion_bridge_feat = BridgeFeatLoss()
        self.criterion_diverse = DivLoss()

    def train(self, epoch, data_loader_source, data_loader_target, source_classes, target_classes,
              optimizer, print_freq=50, train_iters=400, use_xbm=False, stage=0):

        self.criterion_ce = BridgeProbLoss(source_classes + target_classes).cuda()

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        losses_xbm = AverageMeter()
        losses_bridge_prob = AverageMeter()
        losses_bridge_feat = AverageMeter()
        losses_diverse = AverageMeter()
        
        precisions_s = AverageMeter()
        precisions_t = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            source_inputs = data_loader_source.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            s_inputs, s_targets, _ = self._parse_data(source_inputs)
            t_inputs, t_targets, t_indexes = self._parse_data(target_inputs)

            # arrange batch for domain-specific BN
            device_num = torch.cuda.device_count()
            B, C, H, W = s_inputs.size()

            def reshape(inputs):
                return inputs.view(device_num, -1, C, H, W)

            s_inputs, t_inputs = reshape(s_inputs), reshape(t_inputs)
            inputs = torch.cat((s_inputs, t_inputs), 1).view(-1, C, H, W)

            targets = torch.cat((s_targets.view(device_num, -1), t_targets.view(device_num, -1)), 1)
            targets = targets.view(-1)
            # forward
            prob, feats, attention_lam= self._forward(inputs, stage) # attention_lam: [B, 2]
            prob = prob[:, 0:source_classes + target_classes]

            # split feats
            ori_feats = feats.view(device_num, -1, feats.size(-1))
            feats_s, feats_t, feats_mixed = ori_feats.split(ori_feats.size(1) // 3, dim=1)
            ori_feats = torch.cat((feats_s, feats_t), 1).view(-1, ori_feats.size(-1))

            # classification+triplet
            loss_ce, loss_bridge_prob = self.criterion_ce(prob, targets, attention_lam[:,0].detach())
            loss_tri = self.criterion_tri(ori_feats, targets)
            loss_diverse = self.criterion_diverse(attention_lam)

            feats_s = feats_s.contiguous().view(-1, feats.size(-1))
            feats_t = feats_t.contiguous().view(-1, feats.size(-1))
            feats_mixed = feats_mixed.contiguous().view(-1, feats.size(-1))

            loss_bridge_feat = self.criterion_bridge_feat(feats_s, feats_t, feats_mixed, attention_lam)


            # enqueue and dequeue for xbm
            if use_xbm:
                self.xbm.enqueue_dequeue(ori_feats.detach(), targets.detach())
                xbm_feats, xbm_targets = self.xbm.get()
                loss_xbm = self.criterion_tri_xbm(ori_feats, targets, xbm_feats, xbm_targets)
                losses_xbm.update(loss_xbm.item())
                loss = (1.-self.mu1) * loss_ce + loss_tri + loss_xbm + \
                       self.mu1 * loss_bridge_prob + self.mu2 * loss_bridge_feat + self.mu3 * loss_diverse
            else:
                loss = (1.-self.mu1) * loss_ce + loss_tri + \
                       self.mu1 * loss_bridge_prob + self.mu2 * loss_bridge_feat + self.mu3 * loss_diverse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ori_prob = prob.view(device_num, -1, prob.size(-1))
            prob_s, prob_t, _ = ori_prob.split(ori_prob.size(1) // 3, dim=1)

            prob_s, prob_t = prob_s.contiguous(), prob_t.contiguous()
            prec_s, = accuracy(prob_s.view(-1, prob_s.size(-1)).data, s_targets.data)
            prec_t, = accuracy(prob_t.view(-1, prob_s.size(-1)).data, t_targets.data)

            losses.update(loss.item())
            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            losses_bridge_prob.update(loss_bridge_prob.item())
            losses_bridge_feat.update(loss_bridge_feat.item())
            losses_diverse.update(loss_diverse.item())
            
            precisions_s.update(prec_s[0])
            precisions_t.update(prec_t[0])

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:

                if use_xbm:
                    print('Epoch: [{}][{}/{}]\t'
                          'Time {:.3f} ({:.3f}) '
                          'Data {:.3f} ({:.3f}) '
                          'Loss {:.3f} ({:.3f}) '
                          'Loss_ce {:.3f} ({:.3f}) '
                          'Loss_tri {:.3f} ({:.3f}) '
                          'Loss_xbm {:.3f} ({:.3f}) '
                          'Loss_bridge_prob {:.3f} ({:.3f}) '
                          'Loss_bridge_feat {:.3f} ({:.3f}) '
                          'Loss_diverse {:.3f} ({:.3f}) '
                          'Prec_s {:.2%} ({:.2%}) '
                          'Prec_t {:.2%} ({:.2%}) '
                          .format(epoch, i + 1, len(data_loader_target),
                                  batch_time.val, batch_time.avg,
                                  data_time.val, data_time.avg,
                                  losses.val, losses.avg,
                                  losses_ce.val, losses_ce.avg,
                                  losses_tri.val, losses_tri.avg,
                                  losses_xbm.val, losses_xbm.avg,
                                  losses_bridge_prob.val, losses_bridge_prob.avg,
                                  losses_bridge_feat.val, losses_bridge_feat.avg,
                                  losses_diverse.val, losses_diverse.avg,
                                  precisions_s.val, precisions_s.avg,
                                  precisions_t.val, precisions_t.avg
                                  ))
                else:
                    print('Epoch: [{}][{}/{}]\t'
                          'Time {:.3f} ({:.3f}) '
                          'Data {:.3f} ({:.3f}) '
                          'Loss {:.3f} ({:.3f}) '
                          'Loss_ce {:.3f} ({:.3f}) '
                          'Loss_tri {:.3f} ({:.3f}) '
                          'Loss_bridge_prob {:.3f} ({:.3f}) '
                          'Loss_bridge_feat {:.3f} ({:.3f}) '
                          'Loss_diverse {:.3f} ({:.3f}) '
                          'Prec_s {:.2%} ({:.2%}) '
                          'Prec_t {:.2%} ({:.2%}) '
                          .format(epoch, i + 1, len(data_loader_target),
                                  batch_time.val, batch_time.avg,
                                  data_time.val, data_time.avg,
                                  losses.val, losses.avg,
                                  losses_ce.val, losses_ce.avg,
                                  losses_tri.val, losses_tri.avg,
                                  losses_bridge_prob.val, losses_bridge_prob.avg,
                                  losses_bridge_feat.val, losses_bridge_feat.avg,
                                  losses_diverse.val, losses_diverse.avg,
                                  precisions_s.val, precisions_s.avg,
                                  precisions_t.val, precisions_t.avg
                                  ))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs, stage):
        return self.model(inputs, stage=stage)
