#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import datetime
import os
import time
import cv2
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from common.avgmeter import *
from torch.utils.tensorboard import SummaryWriter
from common.sync_batchnorm.batchnorm import convert_model
from modules.scheduler.warmupLR import *
from modules.ioueval import *
from modules.losses.Lovasz_Softmax import Lovasz_softmax
from modules.scheduler.cosine import CosineAnnealingWarmUpRestarts

from tqdm import tqdm



def save_to_log(logdir, logfile, message):
    f = open(logdir + '/' + logfile, "a")
    f.write(message + '\n')
    f.close()
    return


def save_checkpoint(to_save, logdir, suffix=""):
    # Save the weights
    torch.save(to_save, logdir +
               "/SENet" + suffix)


class Trainer():
    def __init__(self, ARCH, DATA, datadir, logdir, path=None):
        # parameters
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.log = logdir
        self.path = path

        self.batch_time_t = AverageMeter()
        self.data_time_t = AverageMeter()
        self.batch_time_e = AverageMeter()
        self.epoch = 0

        # put logger where it belongs

        self.info = {"train_loss": 0,
                     "train_acc": 0,
                     "train_iou": 0,
                     "valid_loss": 0,
                     "valid_acc": 0,
                     "valid_iou": 0,
                     "best_train_iou": 0,
                     "best_val_iou": 0}

        # get the data
        from dataset.kitti.parser import Parser
        self.parser = Parser(root=self.datadir,
                                          train_sequences=self.DATA["split"]["train"], # self.DATA["split"]["valid"] + self.DATA["split"]["train"] if finetune with valid
                                          valid_sequences=self.DATA["split"]["valid"],
                                          test_sequences=None,
                                          labels=self.DATA["labels"],
                                          color_map=self.DATA["color_map"],
                                          learning_map=self.DATA["learning_map"],
                                          learning_map_inv=self.DATA["learning_map_inv"],
                                          sensor=self.ARCH["dataset"]["sensor"],
                                          max_points=self.ARCH["dataset"]["max_points"],
                                          batch_size=self.ARCH["train"]["batch_size"],
                                          workers=self.ARCH["train"]["workers"],
                                          gt=True,
                                          shuffle_train=True)

        # weights for loss (and bias)

        epsilon_w = self.ARCH["train"]["epsilon_w"]
        content = torch.zeros(self.parser.get_n_classes(), dtype=torch.float)
        for cl, freq in DATA["content"].items():
            x_cl = self.parser.to_xentropy(cl)  # map actual class to xentropy class
            content[x_cl] += freq
        self.loss_w = 1 / (content + epsilon_w)  # get weights

#         power_value = 0.25
#         self.loss_w = np.power(self.loss_w, power_value) * np.power(10, 1 - power_value)

        for x_cl, w in enumerate(self.loss_w):  # ignore the ones necessary to ignore
            if DATA["learning_ignore"][x_cl]:
                # don't weigh
                self.loss_w[x_cl] = 0
        print("Loss weights from content: ", self.loss_w.data)

        with torch.no_grad():
            if self.ARCH["train"]["pipeline"] == "hardnet":
                from modules.network.HarDNet import HarDNet
                self.model = HarDNet(self.parser.get_n_classes(), self.ARCH["train"]["aux_loss"])

            if self.ARCH["train"]["pipeline"] == "res":
                from modules.network.ResNet import ResNet_34
                self.model = ResNet_34(self.parser.get_n_classes(), self.ARCH["train"]["aux_loss"])

                def convert_relu_to_softplus(model, act):
                    for child_name, child in model.named_children():
                        if isinstance(child, nn.LeakyReLU):
                            setattr(model, child_name, act)
                        else:
                            convert_relu_to_softplus(child, act)
                if self.ARCH["train"]["act"] == "Hardswish":
                    convert_relu_to_softplus(self.model, nn.Hardswish())
                elif self.ARCH["train"]["act"] == "SiLU":
                    convert_relu_to_softplus(self.model, nn.SiLU())

            if self.ARCH["train"]["pipeline"] == "fid":
                from modules.network.Fid import ResNet_34
                self.model = ResNet_34(self.parser.get_n_classes(), self.ARCH["train"]["aux_loss"])

                if self.ARCH["train"]["act"] == "Hardswish":
                    convert_relu_to_softplus(self.model, nn.Hardswish())
                elif self.ARCH["train"]["act"] == "SiLU":
                    convert_relu_to_softplus(self.model, nn.SiLU())

        save_to_log(self.log, 'model.txt', str(self.model))
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Number of parameters: ", pytorch_total_params/1000000, "M")
        save_to_log(self.log, 'model.txt', "Number of parameters: %.5f M" %(pytorch_total_params/1000000))
        self.tb_logger = SummaryWriter(log_dir=self.log, flush_secs=20)

        # GPU?
        self.gpu = False
        self.multi_gpu = False
        self.n_gpus = 0
        self.model_single = self.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Training in device: ", self.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.n_gpus = 1
            self.model.cuda()
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)  # spread in gpus
            self.model = convert_model(self.model).cuda()  # sync batchnorm
            self.model_single = self.model.module  # single model to get weight names
            self.multi_gpu = True
            self.n_gpus = torch.cuda.device_count()


        self.criterion = nn.NLLLoss(weight=self.loss_w).to(self.device)
        self.ls = Lovasz_softmax(ignore=0).to(self.device)
        from modules.losses.boundary_loss import BoundaryLoss
        self.bd = BoundaryLoss().to(self.device)
        # loss as dataparallel too (more images in batch)
        if self.n_gpus > 1:
            self.criterion = nn.DataParallel(self.criterion).cuda()  # spread in gpus
            self.ls = nn.DataParallel(self.ls).cuda()

#         self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0001, weight_decay=0.0005)
#         from modules.adam_policy import MyLR
#         self.scheduler = MyLR(optimizer=self.optimizer, cycle=30)
#         print(self.optimizer)

        if self.ARCH["train"]["scheduler"] == "consine":
            length = self.parser.get_train_size()
            dict = self.ARCH["train"]["consine"]
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=dict["min_lr"],
                                       momentum=self.ARCH["train"]["momentum"],
                                       weight_decay=self.ARCH["train"]["w_decay"])
            self.scheduler = CosineAnnealingWarmUpRestarts(optimizer=self.optimizer,
                                                           T_0=dict["first_cycle"] * length, T_mult=dict["cycle"],
                                                           eta_max=dict["max_lr"],
                                                           T_up=dict["wup_epochs"]*length, gamma=dict["gamma"])

        else:
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=self.ARCH["train"]["decay"]["lr"],
                                       momentum=self.ARCH["train"]["momentum"],
                                       weight_decay=self.ARCH["train"]["w_decay"])
            steps_per_epoch = self.parser.get_train_size()
            up_steps = int(self.ARCH["train"]["decay"]["wup_epochs"] * steps_per_epoch)
            final_decay = self.ARCH["train"]["decay"]["lr_decay"] ** (1 / steps_per_epoch)
            self.scheduler = warmupLR(optimizer=self.optimizer,
                                      lr=self.ARCH["train"]["decay"]["lr"],
                                      warmup_steps=up_steps,
                                      momentum=self.ARCH["train"]["momentum"],
                                      decay=final_decay)

        if self.path is not None:
            torch.nn.Module.dump_patches = True
            w_dict = torch.load(path + "/SENet_valid_best",
                                map_location=lambda storage, loc: storage)
            self.model.load_state_dict(w_dict['state_dict'], strict=True)
#             self.optimizer.load_state_dict(w_dict['optimizer'])
#             self.epoch = w_dict['epoch'] + 1
#             self.scheduler.load_state_dict(w_dict['scheduler'])
            print("dict epoch:", w_dict['epoch'])
#             self.info = w_dict['info']
            print("info", w_dict['info'])


    def calculate_estimate(self, epoch, iter):
        estimate = int((self.data_time_t.avg + self.batch_time_t.avg) * \
                       (self.parser.get_train_size() * self.ARCH['train']['max_epochs'] - (
                               iter + 1 + epoch * self.parser.get_train_size()))) + \
                   int(self.batch_time_e.avg * self.parser.get_valid_size() * (
                           self.ARCH['train']['max_epochs'] - (epoch)))
        return str(datetime.timedelta(seconds=estimate))

    @staticmethod
    def get_mpl_colormap(cmap_name):
        cmap = plt.get_cmap(cmap_name)
        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)
        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
        return color_range.reshape(256, 1, 3)

    @staticmethod
    def make_log_img(depth, mask, pred, gt, color_fn):
        # input should be [depth, pred, gt]
        # make range image (normalized to 0,1 for saving)
        depth = (cv2.normalize(depth, None, alpha=0, beta=1,
                               norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_32F) * 255.0).astype(np.uint8)
        out_img = cv2.applyColorMap(
            depth, Trainer.get_mpl_colormap('viridis')) * mask[..., None]
        # make label prediction
        pred_color = color_fn((pred * mask).astype(np.int32))
        out_img = np.concatenate([out_img, pred_color], axis=0)
        # make label gt
        gt_color = color_fn(gt)
        out_img = np.concatenate([out_img, gt_color], axis=0)
        return (out_img).astype(np.uint8)

    @staticmethod
    def save_to_log(logdir, logger, info, epoch, w_summary=False, model=None, img_summary=False, imgs=[]):
        # save scalars
        for tag, value in info.items():
            logger.add_scalar(tag, value, epoch)

        # save summaries of weights and biases
        if w_summary and model:
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
                if value.grad is not None:
                    logger.histo_summary(
                        tag + '/grad', value.grad.data.cpu().numpy(), epoch)

        if img_summary and len(imgs) > 0:
            directory = os.path.join(logdir, "predictions")
            if not os.path.isdir(directory):
                os.makedirs(directory)
            for i, img in enumerate(imgs):
                name = os.path.join(directory, str(i) + ".png")
                cv2.imwrite(name, img)

    def train(self):

        self.ignore_class = []
        for i, w in enumerate(self.loss_w):
            if w < 1e-10:
                self.ignore_class.append(i)
                print("Ignoring class ", i, " in IoU evaluation")
        self.evaluator = iouEval(self.parser.get_n_classes(),
                                 self.device, self.ignore_class)
        save_to_log(self.log, 'log.txt', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        if self.path is not None:
            acc, iou, loss, rand_img = self.validate(val_loader=self.parser.get_valid_set(),
                                             model=self.model,
                                             criterion=self.criterion,
                                             evaluator=self.evaluator,
                                             class_func=self.parser.get_xentropy_class_string,
                                             color_fn=self.parser.to_color,
                                             save_scans=self.ARCH["train"]["save_scans"])

        # train for n epochs
        for epoch in range(self.epoch, self.ARCH["train"]["max_epochs"]):
            # train for 1 epoch

            acc, iou, loss = self.train_epoch(train_loader=self.parser.get_train_set(),
                                                           model=self.model,
                                                           criterion=self.criterion,
                                                           optimizer=self.optimizer,
                                                           epoch=epoch,
                                                           evaluator=self.evaluator,
                                                           scheduler=self.scheduler,
                                                           color_fn=self.parser.to_color,
                                                           report=self.ARCH["train"]["report_batch"],
                                                           show_scans=self.ARCH["train"]["show_scans"])


            # update info
            self.info["train_loss"] = loss
            self.info["train_acc"] = acc
            self.info["train_iou"] = iou

            # remember best iou and save checkpoint
            state = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'info': self.info,
                     'scheduler': self.scheduler.state_dict()
                     }
            save_checkpoint(state, self.log, suffix="")
            # save_checkpoint(state, self.log, suffix=""+str(epoch))

            if self.info['train_iou'] > self.info['best_train_iou']:
                save_to_log(self.log, 'log.txt', "Best mean iou in training set so far, save model!")
                print("Best mean iou in training set so far, save model!")
                self.info['best_train_iou'] = self.info['train_iou']
                state = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                         'optimizer': self.optimizer.state_dict(),
                         'info': self.info,
                         'scheduler': self.scheduler.state_dict()
                         }
                save_checkpoint(state, self.log, suffix="_train_best")

            if epoch % self.ARCH["train"]["report_epoch"] == 0:
                # evaluate on validation set
                print("*" * 80)
                acc, iou, loss, rand_img = self.validate(val_loader=self.parser.get_valid_set(),
                                                         model=self.model,
                                                         criterion=self.criterion,
                                                         evaluator=self.evaluator,
                                                         class_func=self.parser.get_xentropy_class_string,
                                                         color_fn=self.parser.to_color,
                                                         save_scans=self.ARCH["train"]["save_scans"])

                # update info
                self.info["valid_loss"] = loss
                self.info["valid_acc"] = acc
                self.info["valid_iou"] = iou

            # remember best iou and save checkpoint
            if self.info['valid_iou'] > self.info['best_val_iou']:
                save_to_log(self.log, 'log.txt', "Best mean iou in validation so far, save model!")
                print("Best mean iou in validation so far, save model!")
                print("*" * 80)
                self.info['best_val_iou'] = self.info['valid_iou']

                # save the weights!
                state = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                         'optimizer': self.optimizer.state_dict(),
                         'info': self.info,
                         'scheduler': self.scheduler.state_dict()
                         }
                save_checkpoint(state, self.log, suffix="_valid_best")

            print("*" * 80)

            # save to log
            Trainer.save_to_log(logdir=self.log,
                                logger=self.tb_logger,
                                info=self.info,
                                epoch=epoch,
                                w_summary=self.ARCH["train"]["save_summary"],
                                model=self.model_single,
                                img_summary=self.ARCH["train"]["save_scans"],
                                imgs=rand_img)
            save_to_log(self.log, 'log.txt', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print('Finished Training')
        save_to_log(self.log, 'log.txt', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        return

    def train_epoch(self, train_loader, model, criterion, optimizer, epoch, evaluator, scheduler, color_fn, report=10,
                    show_scans=False):
        losses = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        update_ratio_meter = AverageMeter()
        bd = AverageMeter()

        # empty the cache to train now
        if self.gpu:
            torch.cuda.empty_cache()

        scaler = torch.cuda.amp.GradScaler()

        # switch to train mode
        model.train()

        end = time.time()
        for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name, _, _, _, _, _, _, _, _, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
            # measure data loading time
            self.data_time_t.update(time.time() - end)
            if not self.multi_gpu and self.gpu:
                in_vol = in_vol.cuda()
            if self.gpu:
                proj_labels = proj_labels.cuda().long()

#                 proj_labels = proj_labels.unsqueeze(1).type(torch.FloatTensor)
#                 from torch.nn import functional as F
#                 [n, c, h, w] = proj_labels.size()
#                 proj_labels_8 = F.interpolate(proj_labels, size=(h//8, w//8), mode='nearest').squeeze(1).cuda().long()
#                 proj_labels_4 = F.interpolate(proj_labels, size=(h//4, w//4), mode='nearest').squeeze(1).cuda().long()
#                 proj_labels_2 = F.interpolate(proj_labels, size=(h//2, w//2), mode='nearest').squeeze(1).cuda().long()
#                 proj_labels = proj_labels.squeeze(1).cuda().long()


            # compute output
            with torch.cuda.amp.autocast():

#                 if self.ARCH["train"]["aux_loss"]:
#                     [output, z2, z4, z8] = model(in_vol)
#                     lamda = self.ARCH["train"]["lamda"]
#                     bdlosss = self.bd(output, proj_labels.long()) + lamda*self.bd(z2, proj_labels_2.long()) + lamda*self.bd(z4, proj_labels_4.long()) + lamda*self.bd(z8, proj_labels_8.long())
#                     loss_m0 = criterion(torch.log(output.clamp(min=1e-8)), proj_labels) + 1.5 * self.ls(output, proj_labels.long())
#                     loss_m2 = criterion(torch.log(z2.clamp(min=1e-8)), proj_labels_2) + 1.5 * self.ls(z2, proj_labels_2.long())
#                     loss_m4 = criterion(torch.log(z4.clamp(min=1e-8)), proj_labels_4) + 1.5 * self.ls(z4, proj_labels_4.long())
#                     loss_m8 = criterion(torch.log(z8.clamp(min=1e-8)), proj_labels_8) + 1.5 * self.ls(z8, proj_labels_8.long())
#                     loss_m = loss_m0 + lamda*loss_m2 + lamda*loss_m4 + lamda*loss_m8 + bdlosss

                if self.ARCH["train"]["aux_loss"]:
                    [output, z2, z4, z8] = model(in_vol)
                    lamda = self.ARCH["train"]["lamda"]
                    bdlosss = self.bd(output, proj_labels.long()) + lamda*self.bd(z2, proj_labels.long()) + lamda*self.bd(z4, proj_labels.long()) + lamda*self.bd(z8, proj_labels.long())
                    loss_m0 = criterion(torch.log(output.clamp(min=1e-8)), proj_labels) + 1.5 * self.ls(output, proj_labels.long())
                    loss_m2 = criterion(torch.log(z2.clamp(min=1e-8)), proj_labels) + 1.5 * self.ls(z2, proj_labels.long())
                    loss_m4 = criterion(torch.log(z4.clamp(min=1e-8)), proj_labels) + 1.5 * self.ls(z4, proj_labels.long())
                    loss_m8 = criterion(torch.log(z8.clamp(min=1e-8)), proj_labels) + 1.5 * self.ls(z8, proj_labels.long())
                    loss_m = loss_m0 + lamda*loss_m2 + lamda*loss_m4 + lamda*loss_m8 + bdlosss
                else:
                    output = model(in_vol)
                    bdlosss = self.bd(output, proj_labels.long())
                    loss_m = criterion(torch.log(output.clamp(min=1e-8)), proj_labels) + 1.5 * self.ls(output, proj_labels.long()) + bdlosss

            optimizer.zero_grad()

#             if self.n_gpus > 1:
#                 idx = torch.ones(self.n_gpus).cuda()
#                 loss_m.backward(idx)
#             else:
#                 loss_m.backward()
#             optimizer.step()

            scaler.scale(loss_m).backward()
            scaler.step(optimizer)
            scaler.update()

            # measure accuracy and record loss
            loss = loss_m.mean()
            with torch.no_grad():
                evaluator.reset()
                argmax = output.argmax(dim=1)
                evaluator.addBatch(argmax, proj_labels)
                accuracy = evaluator.getacc()
                jaccard, class_jaccard = evaluator.getIoU()

            losses.update(loss.item(), in_vol.size(0))
            acc.update(accuracy.item(), in_vol.size(0))
            iou.update(jaccard.item(), in_vol.size(0))
            bd.update(bdlosss.item(), in_vol.size(0))

            # measure elapsed time
            self.batch_time_t.update(time.time() - end)
            end = time.time()

            # get gradient updates and weights, so I can print the relationship of
            # their norms
            update_ratios = []
            for g in self.optimizer.param_groups:
                lr = g["lr"]

            if show_scans:
                if i % self.ARCH["train"]["save_batch"] == 0:
                    # get the first scan in batch and project points
                    mask_np = proj_mask[0].cpu().numpy()
                    depth_np = in_vol[0][0].cpu().numpy()
                    pred_np = argmax[0].cpu().numpy()
                    gt_np = proj_labels[0].cpu().numpy()
                    out = Trainer.make_log_img(depth_np, mask_np, pred_np, gt_np, color_fn)

                    directory = os.path.join(self.log, "train-predictions")
                    if not os.path.isdir(directory):
                        os.makedirs(directory)
                    name = os.path.join(directory, str(i) + ".png")
                    cv2.imwrite(name, out)


            if i % self.ARCH["train"]["report_batch"] == 0:
                print('Lr: {lr:.3e} | '
                      'Epoch: [{0}][{1}/{2}] | '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                      'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                      'Bd {bd.val:.4f} ({bd.avg:.4f}) | '
                      'acc {acc.val:.3f} ({acc.avg:.3f}) | '
                      'IoU {iou.val:.3f} ({iou.avg:.3f}) | [{estim}]'.format(
                    epoch, i, len(train_loader), batch_time=self.batch_time_t,
                    data_time=self.data_time_t, loss=losses, bd=bd, acc=acc, iou=iou, lr=lr,
                    estim=self.calculate_estimate(epoch, i)))

                save_to_log(self.log, 'log.txt', 'Lr: {lr:.3e} | '
                                                 'Epoch: [{0}][{1}/{2}] | '
                                                 'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                                                 'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                                                 'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                                                 'Bd {bd.val:.4f} ({bd.avg:.4f}) | '
                                                 'acc {acc.val:.3f} ({acc.avg:.3f}) | '
                                                 'IoU {iou.val:.3f} ({iou.avg:.3f}) | [{estim}]'.format(
                    epoch, i, len(train_loader), batch_time=self.batch_time_t,
                    data_time=self.data_time_t, loss=losses, bd=bd, acc=acc, iou=iou, lr=lr,
                    estim=self.calculate_estimate(epoch, i)))
            # step scheduler
            scheduler.step()
#         scheduler.step()
        return acc.avg, iou.avg, losses.avg

    def validate(self, val_loader, model, criterion, evaluator, class_func, color_fn, save_scans):
        losses = AverageMeter()
        jaccs = AverageMeter()
        wces = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        rand_imgs = []

        # switch to evaluate mode
        model.eval()
        evaluator.reset()

        # empty the cache to infer in high res
        if self.gpu:
            torch.cuda.empty_cache()

        with torch.no_grad():
            end = time.time()
            for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name, _, _, _, _, _, _, _, _, _) in tqdm(enumerate(val_loader), total=len(val_loader)):
                if not self.multi_gpu and self.gpu:
                    in_vol = in_vol.cuda()
                    proj_mask = proj_mask.cuda()
                if self.gpu:
                    proj_labels = proj_labels.cuda(non_blocking=True).long()

                # compute output
                if self.ARCH["train"]["aux_loss"]:
                    [output, z2, z4, z8] = model(in_vol)
                else:
                    output = model(in_vol)

                log_out = torch.log(output.clamp(min=1e-8))
                jacc = self.ls(output, proj_labels)
                wce = criterion(log_out, proj_labels)
                loss = wce + jacc

                # measure accuracy and record loss
                argmax = output.argmax(dim=1)
                evaluator.addBatch(argmax, proj_labels)
                losses.update(loss.mean().item(), in_vol.size(0))
                jaccs.update(jacc.mean().item(),in_vol.size(0))


                wces.update(wce.mean().item(),in_vol.size(0))



                if save_scans:
                    # get the first scan in batch and project points
                    mask_np = proj_mask[0].cpu().numpy()
                    depth_np = in_vol[0][0].cpu().numpy()
                    pred_np = argmax[0].cpu().numpy()
                    gt_np = proj_labels[0].cpu().numpy()
                    out = Trainer.make_log_img(depth_np,
                                               mask_np,
                                               pred_np,
                                               gt_np,
                                               color_fn)
                    rand_imgs.append(out)

                # measure elapsed time
                self.batch_time_e.update(time.time() - end)
                end = time.time()

            accuracy = evaluator.getacc()
            jaccard, class_jaccard = evaluator.getIoU()
            acc.update(accuracy.item(), in_vol.size(0))
            iou.update(jaccard.item(), in_vol.size(0))

            print('Validation set:\n'
                  'Time avg per batch {batch_time.avg:.3f}\n'
                  'Loss avg {loss.avg:.4f}\n'
                  'Jaccard avg {jac.avg:.4f}\n'
                  'WCE avg {wces.avg:.4f}\n'
                  'Acc avg {acc.avg:.3f}\n'
                  'IoU avg {iou.avg:.3f}'.format(batch_time=self.batch_time_e,
                                                 loss=losses,
                                                 jac=jaccs,
                                                 wces=wces,
                                                 acc=acc, iou=iou))

            save_to_log(self.log, 'log.txt', 'Validation set:\n'
                                             'Time avg per batch {batch_time.avg:.3f}\n'
                                             'Loss avg {loss.avg:.4f}\n'
                                             'Jaccard avg {jac.avg:.4f}\n'
                                             'WCE avg {wces.avg:.4f}\n'
                                             'Acc avg {acc.avg:.3f}\n'
                                             'IoU avg {iou.avg:.3f}'.format(batch_time=self.batch_time_e,
                                                                            loss=losses,
                                                                            jac=jaccs,
                                                                            wces=wces,
                                                                            acc=acc, iou=iou))
            # print also classwise
            for i, jacc in enumerate(class_jaccard):
                print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                    i=i, class_str=class_func(i), jacc=jacc))
                save_to_log(self.log, 'log.txt', 'IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                    i=i, class_str=class_func(i), jacc=jacc))
                self.info["valid_classes/" + class_func(i)] = jacc


        return acc.avg, iou.avg, losses.avg, rand_imgs
