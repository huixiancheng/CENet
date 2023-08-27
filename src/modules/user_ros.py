#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
import os
import numpy as np
from postproc.KNN import KNN

import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointField, PointCloud2
from sensor_msgs import point_cloud2
import sensor_msgs.point_cloud2 as pcl2
import numpy as np

import cv2

class User():
    def __init__(self, ARCH, DATA, datadir, logdir, modeldir,split):
        # parameters
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.logdir = logdir
        self.modeldir = modeldir
        self.split = split

        # ROS INIT
        rospy.init_node('pointcloud_inference', anonymous=True)
        self.header = Header()
        self.header.stamp = rospy.Time.now()
        self.header.frame_id = 'base_link'
        self.fields =[PointField('x',  0, 7, 1), # PointField.FLOAT32 = 7
                 PointField('y',  4, 7, 1),
                 PointField('z',  8, 7, 1),
                 PointField('intensity',  12, 7, 1),
                 PointField('t', 16, 6, 1),
                 PointField('ring', 20, 2, 1)]
        
        self.point_type = np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('intensity', np.float32),
            ('t', np.uint32),
            ('ring', np.uint8)
        ])

        # get the data
        from dataset.kitti.parser_ros import Parser
        self.parser = Parser(root=self.datadir,
                            #test_sequences=self.DATA["split"]["test"],
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
        w_dict = torch.load(modeldir + "/SalsaNext_valid_best", map_location=lambda storage, loc: storage)
        self.model.load_state_dict(w_dict['state_dict'], strict=True)
        # use knn post processing?
        self.post = None
        
        if self.ARCH["post"]["KNN"]["use"]:
            self.post = KNN(self.ARCH["post"]["KNN"]["params"], self.parser.get_n_classes())
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

    def listener(self):
        # Subscribe to the input PointCloud2 topic
        sc_lio_sam_global_map = '/sc_lio_sam/map_global'
        #sc_lio_sam_local_map = '/sc_lio_sam/map_local'
        
        self.model.eval()
        # empty the cache to infer in high res
        if self.gpu:
            torch.cuda.empty_cache()
        
        ouster_points = '/ouster/points'
        rospy.Subscriber(ouster_points, PointCloud2, self.infer)

        # Create a publisher for the output PointCloud2 topic
        #self.pub = rospy.Publisher('/semantic_points', PointCloud2, queue_size=10)

        rospy.spin()

    def infer(self, data):
        #rospy.loginfo('Received a PointCloud2 message')
        self.header.stamp = data.header.stamp
        
        cnn = []
        knn = []
        loader=self.parser.get_test_set(data)
        to_orig_fn=self.parser.to_original

        self.infer_subset(loader, to_orig_fn, cnn=cnn, knn=knn)
    
        print("Mean CNN inference time:{}\t std:{}".format(np.mean(cnn), np.std(cnn)))
        print("Mean KNN inference time:{}\t std:{}".format(np.mean(knn), np.std(knn)))
        print("Total Frames:{}".format(len(cnn)))
        #print("Finished Infering")

        return

    def publish(self, labels):
        #self.header.stamp = rospy.Time.now() #timestamp
        semantic_points = self.parser.get_scan()
        #semantic_points = semantic_points.reshape(64, 1024)
        print(semantic_points)
        semantic_points['intensity'] = labels

        labeled_cloud = pcl2.create_cloud(self.header, self.fields, semantic_points)
        labeled_cloud.is_dense = True  # Added line
        self.pub.publish(labeled_cloud)

    def infer_subset(self, loader, to_orig_fn,cnn,knn):
        # switch to evaluate mode
        #self.model.eval()
        #total_time=0
        #total_frames=0
        
        # empty the cache to infer in high res
        #if self.gpu:
        #    torch.cuda.empty_cache()
        
        with torch.no_grad():
            proj_in, proj_mask, _, _, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints = loader
            # first cut to rela size (batch size one allows it)

            p_x = p_x[:npoints]
            p_y = p_y[:npoints]
            proj_range = proj_range[:npoints]
            unproj_range = unproj_range[:npoints]  # normalized range data ... 
            print(unproj_range)

            proj_range_np = proj_range.numpy()
            cv_image = (proj_range_np)#.astype(np.uint16)
            print(cv_image.shape)
            cv2.imshow("Model image", cv_image)
            key = cv2.waitKey(1000) & 0xFF
            cv2.destroyAllWindows()

            '''
            # Convert tensors to numpy
            p_x_np = p_x.numpy()   
            p_y_np = p_y.numpy()
            proj_range_np = proj_range.numpy()

            # Convert the intensity values (proj_range) to a grayscale image
            # You might need to normalize the values to be in the range [0, 255]
            cv_image = (proj_range_np)#.astype(np.uint16)
            
            print(cv_image.shape)
            cv2.imshow("Model image", cv_image)

            key = cv2.waitKey(1) & 0xFF
            cv2.destroyAllWindows()
            
            # path_seq = path_seq[0]
            # path_name = path_name[0]

            print(f"p_x.shape: {p_x.shape}, p_y.shape: {p_y.shape}, proj_range.shape: {proj_range.shape}")
            print(f"unproj_range.shape: {unproj_range.shape}, proj_in: {proj_in.shape}")
            '''

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
                    [proj_output, x_2, x_3, x_4] = self.model(proj_in.unsqueeze(0))
            else:
                with torch.cuda.amp.autocast(enabled=True):
                    proj_output = self.model(proj_in)

            proj_argmax = proj_output[0].argmax(dim=0)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            res = time.time() - end
    #        print("Network seq", path_seq, "scan", path_name, "in", res, "sec")
            end = time.time()
            cnn.append(res)

            if self.post:
                # knn postproc
                unproj_argmax = self.post(proj_range, unproj_range, proj_argmax, p_x, p_y)
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
            
            #print(f"KNN Infered point cloud range view in {res} sec")
            knn.append(res)
            end = time.time()

            # save scan
            # get the first scan in batch and project scan
            pred_np = unproj_argmax.cpu().numpy()
            pred_np = pred_np.reshape((-1)).astype(np.int32)

            # map to original label
            pred_np = to_orig_fn(pred_np)
            #print(f"predictions numpy:\n {pred_np}")

            self.publish(pred_np)