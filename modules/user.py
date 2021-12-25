#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
import os
import numpy as np
from postproc.KNN import KNN


class User():
  def __init__(self, ARCH, DATA, datadir, logdir, modeldir,split):
    # parameters
    self.ARCH = ARCH
    self.DATA = DATA
    self.datadir = datadir
    self.logdir = logdir
    self.modeldir = modeldir
    self.split = split

    # get the data
    from dataset.kitti.parser import Parser
    self.parser = Parser(root=self.datadir,
                                      train_sequences=self.DATA["split"]["train"],
                                      valid_sequences=self.DATA["split"]["valid"],
                                      test_sequences=self.DATA["split"]["test"],
                                      labels=self.DATA["labels"],
                                      color_map=self.DATA["color_map"],
                                      learning_map=self.DATA["learning_map"],
                                      learning_map_inv=self.DATA["learning_map_inv"],
                                      sensor=self.ARCH["dataset"]["sensor"],
                                      max_points=self.ARCH["dataset"]["max_points"],
                                      batch_size=1,
                                      workers=self.ARCH["train"]["workers"],
                                      gt=True,
                                      shuffle_train=False)

    # concatenate the encoder and the head
    with torch.no_grad():
        torch.nn.Module.dump_patches = True
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

#     print(self.model)
    w_dict = torch.load(modeldir + "/SENet_valid_best",
                        map_location=lambda storage, loc: storage)
    self.model.load_state_dict(w_dict['state_dict'], strict=True)
    # use knn post processing?
    self.post = None
    if self.ARCH["post"]["KNN"]["use"]:
      self.post = KNN(self.ARCH["post"]["KNN"]["params"],
                      self.parser.get_n_classes())
    print(self.parser.get_n_classes())

    # GPU?
    self.gpu = False
    self.model_single = self.model
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Infering in device: ", self.device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      cudnn.benchmark = True
      cudnn.fastest = True
      self.gpu = True
      self.model.cuda()

  def infer(self):
    cnn = []
    knn = []
    if self.split == None:

        self.infer_subset(loader=self.parser.get_train_set(),
                          to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)

        # do valid set
        self.infer_subset(loader=self.parser.get_valid_set(),
                          to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
        # do test set
        self.infer_subset(loader=self.parser.get_test_set(),
                          to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)


    elif self.split == 'valid':
        self.infer_subset(loader=self.parser.get_valid_set(),
                        to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
    elif self.split == 'train':
        self.infer_subset(loader=self.parser.get_train_set(),
                        to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
    else:
        self.infer_subset(loader=self.parser.get_test_set(),
                        to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
    print("Mean CNN inference time:{}\t std:{}".format(np.mean(cnn), np.std(cnn)))
    print("Mean KNN inference time:{}\t std:{}".format(np.mean(knn), np.std(knn)))
    print("Total Frames:{}".format(len(cnn)))
    print("Finished Infering")

    return

  def infer_subset(self, loader, to_orig_fn,cnn,knn):
    # switch to evaluate mode

    self.model.eval()
    total_time=0
    total_frames=0
    # empty the cache to infer in high res
    if self.gpu:
      torch.cuda.empty_cache()

    with torch.no_grad():
      for i, (proj_in, proj_mask, _, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(loader):
        # first cut to rela size (batch size one allows it)
        p_x = p_x[0, :npoints]
        p_y = p_y[0, :npoints]
        proj_range = proj_range[0, :npoints]
        unproj_range = unproj_range[0, :npoints]
        path_seq = path_seq[0]
        path_name = path_name[0]

        if self.gpu:
          proj_in = proj_in.cuda()
          p_x = p_x.cuda()
          p_y = p_y.cuda()
          if self.post:
            proj_range = proj_range.cuda()
            unproj_range = unproj_range.cuda()
        end = time.time()

        if self.ARCH["train"]["aux_loss"]:
            with torch.cuda.amp.autocast(enabled=True):
                [proj_output, x_2, x_3, x_4] = self.model(proj_in)
        else:
            with torch.cuda.amp.autocast(enabled=True):
                proj_output = self.model(proj_in)

        proj_argmax = proj_output[0].argmax(dim=0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        res = time.time() - end
        print("Network seq", path_seq, "scan", path_name,
              "in", res, "sec")
        end = time.time()
        cnn.append(res)

        if self.post:
            # knn postproc
            unproj_argmax = self.post(proj_range,
                                      unproj_range,
                                      proj_argmax,
                                      p_x,
                                      p_y)
#             # nla postproc
#             proj_unfold_range, proj_unfold_pre = NN_filter(proj_range, proj_argmax)
#             proj_unfold_range=proj_unfold_range.cpu().numpy()
#             proj_unfold_pre=proj_unfold_pre.cpu().numpy()
#             unproj_range = unproj_range.cpu().numpy()
#             #  Check this part. Maybe not correct (Low speed caused by for loop)
#             #  Just simply change from
#             #  https://github.com/placeforyiming/IROS21-FIDNet-SemanticKITTI/blob/7f90b45a765b8bba042b25f642cf12d8fccb5bc2/semantic_inference.py#L177-L202
#             for jj in range(len(p_x)):
#                 py, px = p_y[jj].cpu().numpy(), p_x[jj].cpu().numpy()
#                 if unproj_range[jj] == proj_range[py, px]:
#                     unproj_argmax = proj_argmax[py, px]
#                 else:
#                     potential_label = proj_unfold_pre[0, :, py, px]
#                     potential_range = proj_unfold_range[0, :, py, px]
#                     min_arg = np.argmin(abs(potential_range - unproj_range[jj]))
#                     unproj_argmax = potential_label[min_arg]

        else:
            # put in original pointcloud using indexes
            unproj_argmax = proj_argmax[p_y, p_x]

        # measure elapsed time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        res = time.time() - end
        print("KNN Infered seq", path_seq, "scan", path_name,
              "in", res, "sec")
        knn.append(res)
        end = time.time()

        # save scan
        # get the first scan in batch and project scan
        pred_np = unproj_argmax.cpu().numpy()
        pred_np = pred_np.reshape((-1)).astype(np.int32)

        # map to original label
        pred_np = to_orig_fn(pred_np)

        # save scan
        path = os.path.join(self.logdir, "sequences",
                            path_seq, "predictions", path_name)
        pred_np.tofile(path)
