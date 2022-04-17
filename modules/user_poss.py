#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch.nn as nn
from common.avgmeter import *
import torch.backends.cudnn as cudnn
import time
import os

from postproc.KNN import KNN
from modules.ioueval import *

def save_to_log(logdir,logfile,message):
    f = open(logdir+'/'+logfile, "a")
    f.write(message+'\n')
    f.close()
    return

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
    from dataset.poss.parser import Parser
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

    class_ignore = DATA["learning_ignore"]
    self.ignore = []
    for cl, ign in class_ignore.items():
        if ign:
            x_cl = int(cl)
            self.ignore.append(x_cl)
            print("Ignoring xentropy class ", x_cl, " in IoU evaluation")

    # concatenate the encoder and the head
    with torch.no_grad():
        torch.nn.Module.dump_patches = True
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

    w_dict = torch.load(modeldir + "/SENet_valid_best",
                        map_location=lambda storage, loc: storage)
    self.model.load_state_dict(w_dict['state_dict'], strict=True)

    self.post = None
    if self.ARCH["post"]["KNN"]["use"]:
      print(self.ARCH["post"]["KNN"]["params"])
      self.post = KNN(self.ARCH["post"]["KNN"]["params"],
                      self.parser.get_n_classes())


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
    acc = AverageMeter()
    iou = AverageMeter()

    self.model.eval()

    self.evaluator = iouEval(self.parser.get_n_classes(),
                             self.device, self.ignore)
    self.evaluator.reset()

    self.evaluator2 = iouEval(self.parser.get_n_classes(),
                             self.device, self.ignore)
    self.evaluator2.reset()

    total_time=0
    total_frames=0
    # empty the cache to infer in high res
    if self.gpu:
      torch.cuda.empty_cache()

    with torch.no_grad():
#       end = time.time()

        proj_y = torch.full([40, 1800], 0, dtype=torch.long)
        proj_x = torch.full([40, 1800], 0, dtype=torch.long)
        for i in range(proj_y.size(0)):
          proj_y[i, :] = i
        for i in range(proj_x.size(1)):
          proj_x[:, i] = i

        proj_y = proj_y.reshape([40 * 1800])
        proj_x = proj_x.reshape([40 * 1800])
        proj_x = proj_x.cuda()
        proj_y = proj_y.cuda()

        for i, (proj_in, proj_labels, tags, unlabels, path_seq, path_name, proj_range, unresizerange, unproj_range, _, _) in enumerate(loader):
                # first cut to rela size (batch size one allows it)

                proj_in = proj_in.cuda()
                unlabels = unlabels.cuda()
                path_seq = path_seq[0]
                path_name = path_name[0]
                if self.post:
                    proj_range = proj_range[0].cuda()
                    unproj_range = unproj_range[0].cuda()

                end = time.time()
                if self.ARCH["train"]["aux_loss"]:
                    [proj_output, x_2, x_3, x_4] = self.model(proj_in)
                else:
                    proj_output = self.model(proj_in)
                proj_argmax = proj_output[0].argmax(dim=0)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                res = time.time() - end
                cnn.append(res)
                print("Network seq", path_seq, "scan", path_name,
                      "in", res, "sec")
                end = time.time()

                self.evaluator.addBatch(proj_argmax, proj_labels.squeeze().cpu().numpy())

                # proj_argmax = torch.reshape(proj_argmax, [40*1800])
        #         print(unresizerange.shape, unproj_range.shape, proj_x.shape, proj_y.shape)
                if self.post:
                  unproj_argmax = self.post(proj_range,
                                            unproj_range,
                                            proj_argmax,
                                            proj_x,
                                            proj_y)


                self.evaluator2.addBatch(unproj_argmax, unlabels.squeeze().cpu().numpy())

                unproj_argmax = unproj_argmax[tags.squeeze().cpu().numpy()]
        #         print(unproj_argmax.shape)
                # measure elapsed time
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                res = time.time() - end
                print("KNN Infered seq", path_seq, "scan", path_name,
                      "in", res, "sec")
                knn.append(res)
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

        accuracy = self.evaluator.getacc()
        jaccard, class_jaccard = self.evaluator.getIoU()

        print("No KNN results:")
        print('Acc avg {acc:.3f}\n'
                  'IoU avg {iou:.3f}'.format(acc=accuracy, iou=jaccard))
        save_to_log(self.logdir, 'pred.txt', "No KNN results:")
        save_to_log(self.modeldir,'pred.txt','{split} set:\n'
          'Acc avg {m_accuracy:.3f}\n'
          'IoU avg {m_jaccard:.3f}'.format(split=self.split,
                                           m_accuracy=accuracy,
                                           m_jaccard=jaccard))

        class_func = self.parser.get_xentropy_class_string
        for i, jacc in enumerate(class_jaccard):
          print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
              i=i, class_str=class_func(i), jacc=jacc))
          save_to_log(self.modeldir, 'pred.txt', 'IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                i=i, class_str=class_func(i), jacc=jacc))

        print("**"*10)
        accuracy = self.evaluator2.getacc()
        jaccard, class_jaccard = self.evaluator2.getIoU()

        print("KNN results:")
        print('Acc avg {acc:.3f}\n'
            'IoU avg {iou:.3f}'.format(acc=accuracy, iou=jaccard))
        save_to_log(self.logdir, 'pred.txt', "KNN results:")
        save_to_log(self.modeldir,'pred.txt','{split} set:\n'
          'Acc avg {m_accuracy:.3f}\n'
          'IoU avg {m_jaccard:.3f}'.format(split=self.split,
                                           m_accuracy=accuracy,
                                           m_jaccard=jaccard))

        class_func = self.parser.get_xentropy_class_string
        for i, jacc in enumerate(class_jaccard):
          print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
              i=i, class_str=class_func(i), jacc=jacc))
          save_to_log(self.modeldir, 'pred.txt', 'IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                i=i, class_str=class_func(i), jacc=jacc))

