#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import time

import numpy as np
import math
import random
from scipy.spatial.transform import Rotation as R
import cv2

class LaserScan:
    """Class that contains LaserScan with x,y,z,r"""
    EXTENSIONS_SCAN = ['.bin']

    def __init__(self, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0,DA=False,flip_sign=False,rot=False,drop_points=False):
        self.project = project
        self.proj_H = H
        self.proj_W = W
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down
        self.DA = DA
        self.flip_sign = flip_sign
        self.rot = rot
        self.drop_points = drop_points

        self.reset()

    def reset(self):
        """ Reset scan members. """
        self.points = np.zeros((0, 3), dtype=np.float32)  # [m, 3]: x, y, z
        self.remissions = np.zeros((0, 1), dtype=np.float32)  # [m ,1]: remission

        # projected range image - [H,W] range (-1 is no data)
        self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)

        # unprojected range (list of depths for each point)
        self.unproj_range = np.zeros((0, 1), dtype=np.float32)

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                                dtype=np.float32)

        # projected remission - [H,W] intensity (-1 is no data)
        self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                      dtype=np.float32)

        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                                dtype=np.int32)

        # for each point, where it is in the range image
        self.proj_x = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: y

        # mask containing for each pixel, if it contains a point or not
        self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                                  dtype=np.int32)  # [H,W] mask

    def size(self):
        """ Return the size of the point cloud. """
        return self.points.shape[0]

    def __len__(self):
        return self.size()

    def open_scan(self, filename):
        """ Open raw scan and fill in attributes
        """
        # reset just in case there was an open structure
        self.reset()

        # check filename is string
        if not isinstance(filename, str):
            raise TypeError("Filename should be string type, "
                            "but was {type}".format(type=str(type(filename))))

        # check extension is a laserscan
        if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
            raise RuntimeError("Filename extension is not valid scan file.")

        # if all goes well, open pointcloud
        scan = np.fromfile(filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))

        # put in attribute
        points = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]  # get remission
        if self.drop_points is not False:
            self.points_to_drop = np.random.randint(0, len(points)-1,int(len(points)*self.drop_points))
            points = np.delete(points,self.points_to_drop,axis=0)
            remissions = np.delete(remissions,self.points_to_drop)

        self.set_points(points, remissions)

    def set_points(self, points, remissions=None):
        """ Set scan attributes (instead of opening from file)
        """
        # reset just in case there was an open structure
        self.reset()

        # check scan makes sense
        if not isinstance(points, np.ndarray):
            raise TypeError("Scan should be numpy array")

        # check remission makes sense
        if remissions is not None and not isinstance(remissions, np.ndarray):
            raise TypeError("Remissions should be numpy array")

        # put in attribute
        self.points = points  # get
        if self.flip_sign:
            self.points[:, 1] = -self.points[:, 1]
        if self.DA:
            jitter_x = random.uniform(-5, 5)
            jitter_y = random.uniform(-3, 3)
            jitter_z = random.uniform(-1, 0)
            self.points[:, 0] += jitter_x
            self.points[:, 1] += jitter_y
            self.points[:, 2] += jitter_z
        if self.rot:
            euler_angle = np.random.normal(0, 90, 1)[0]  # 40
            r = np.array(R.from_euler('zyx', [[euler_angle, 0, 0]], degrees=True).as_matrix())
            r_t = r.transpose()
            self.points = self.points.dot(r_t)
            self.points = np.squeeze(self.points)
        if remissions is not None:
            self.remissions = remissions  # get remission
            #if self.DA:
            #    self.remissions = self.remissions[::-1].copy()
        else:
            self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

        # if projection is wanted, then do it and fill in the structure
        if self.project:
            self.do_range_projection()


#         fov_up = 3.0
#         fov_down = -25.0
#         self.fov_up = fov_up
#         self.fov_down = fov_down
#         fov_up = fov_up / 180.0 * np.pi  # field of view up in rad
#         fov_down = fov_down / 180.0 * np.pi  # field of view down in rad
#         fov = abs(fov_down) + abs(fov_up)
#         zero_matrix = np.zeros((self.proj_H, self.proj_W))
#         one_matrix = np.ones((self.proj_H, self.proj_W))

#         self.theta_channel = np.zeros((self.proj_H, self.proj_W))
#         self.phi_channel = np.zeros((self.proj_H, self.proj_W))
#         for i in range(self.proj_H):
#             for j in range(self.proj_W):
#                 self.theta_channel[i, j] = np.pi * (float(j + 0.5) / self.proj_W * 2 - 1)
#                 self.phi_channel[i, j] = (1 - float(i + 0.5) / self.proj_H) * fov - abs(fov_down)
#         self.R_theta = [np.cos(self.theta_channel), -np.sin(self.theta_channel), zero_matrix,
#                         np.sin(self.theta_channel), np.cos(self.theta_channel), zero_matrix, zero_matrix,
#                         zero_matrix, one_matrix]
#         self.R_theta = np.asarray(self.R_theta)
#         self.R_theta = np.transpose(self.R_theta, (1, 2, 0))
#         self.R_theta = np.reshape(self.R_theta, [self.proj_H, self.proj_W, 3, 3])
#         self.R_phi = [np.cos(self.phi_channel), zero_matrix, -np.sin(self.phi_channel), zero_matrix, one_matrix,
#                       zero_matrix, np.sin(self.phi_channel), zero_matrix, np.cos(self.phi_channel)]
#         self.R_phi = np.asarray(self.R_phi)
#         self.R_phi = np.transpose(self.R_phi, (1, 2, 0))
#         self.R_phi = np.reshape(self.R_phi, [self.proj_H, self.proj_W, 3, 3])
#         self.R_theta_phi = np.matmul(self.R_theta, self.R_phi)

#         normal_image = self.calculate_normal(self.fill_spherical(self.proj_range))
#         self.normal_image = normal_image * np.transpose([self.proj_mask, self.proj_mask, self.proj_mask],
#                                                    [1, 2, 0])
    def do_fd_projection(self):
      """ Project a pointcloud into a spherical projection image.projection.
          Function takes no arguments because it can be also called externally
          if the value of the constructor was not set (in case you change your
          mind about wanting the projection)
      """
      # laser parameters
      depth = np.linalg.norm(self.points, 2, axis=1)
      # get scan components
      scan_x = self.points[:, 0]
      scan_y = self.points[:, 1]
      scan_z = self.points[:, 2]

      yaw = -np.arctan2(scan_y, -scan_x)
      proj_x = 0.5 * (yaw / np.pi + 1.0)
      new_raw = np.nonzero((proj_x[1:] < 0.2) * (proj_x[:-1] > 0.8))[0] + 1

      proj_y = np.zeros_like(proj_x)
      proj_y[new_raw] = 1
      proj_y = np.cumsum(proj_y)
      proj_x = proj_x * self.proj_W - 0.001
        
      proj_x = np.floor(proj_x)
      proj_x = np.minimum(self.proj_W - 1, proj_x)
      proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
      self.proj_x = np.copy(proj_x)  # store a copy in orig order

      proj_y = np.floor(proj_y)
      proj_y = np.minimum(self.proj_H - 1, proj_y)
      proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
      self.proj_y = np.copy(proj_y)  # stope a copy in original order
    # stope a copy in original order

      self.unproj_range = np.copy(depth)  # copy of depth in original order

      # order in decreasing depth
      indices = np.arange(depth.shape[0])
      order = np.argsort(depth)[::-1]
      depth = depth[order]
      indices = indices[order]
      points = self.points[order]
      remission = self.remissions[order]
      proj_y = proj_y[order]
      proj_x = proj_x[order]

      # assing to images
      self.proj_range[proj_y, proj_x] = depth
      self.proj_xyz[proj_y, proj_x] = points
      self.proj_remission[proj_y, proj_x] = remission
      self.proj_idx[proj_y, proj_x] = indices
      self.proj_mask = (self.proj_idx > 0).astype(np.float32)
    
    def do_range_projection(self):
        """ Project a pointcloud into a spherical projection image.projection.
            Function takes no arguments because it can be also called externally
            if the value of the constructor was not set (in case you change your
            mind about wanting the projection)
        """
        # laser parameters
        fov_up = self.proj_fov_up / 180.0 * np.pi  # field of view up in rad
        fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # get depth of all points
        depth = np.linalg.norm(self.points, 2, axis=1)

        # get scan components
        scan_x = self.points[:, 0]
        scan_y = self.points[:, 1]
        scan_z = self.points[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.proj_W  # in [0.0, W]
        proj_y *= self.proj_H  # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
        self.proj_x = np.copy(proj_x)  # store a copy in orig order

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
        self.proj_y = np.copy(proj_y)  # stope a copy in original order

        # copy of depth in original order
        self.unproj_range = np.copy(depth)

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = self.points[order]
        remission = self.remissions[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # assing to images
        self.proj_range[proj_y, proj_x] = depth
        self.proj_xyz[proj_y, proj_x] = points
        self.proj_remission[proj_y, proj_x] = remission
        self.proj_idx[proj_y, proj_x] = indices
        self.proj_mask = (self.proj_idx > 0).astype(np.int32)

    def fill_spherical(self, range_image):
        # fill in spherical image for calculating normal vector
        height, width = np.shape(range_image)[:2]
        value_mask = np.asarray(1.0 - np.squeeze(range_image) > 0.1).astype(np.uint8)
        dt, lbl = cv2.distanceTransformWithLabels(value_mask, cv2.DIST_L1, 5, labelType=cv2.DIST_LABEL_PIXEL)

        with_value = np.squeeze(range_image) > 0.1

        depth_list = np.squeeze(range_image)[with_value]

        label_list = np.reshape(lbl, [1, height * width])
        depth_list_all = depth_list[label_list - 1]

        depth_map = np.reshape(depth_list_all, (height, width))

        depth_map = cv2.GaussianBlur(depth_map, (7, 7), 0)
        depth_map = range_image * with_value + depth_map * (1 - with_value)
        return depth_map

    def calculate_normal(self, range_image):

        one_matrix = np.ones((self.proj_H, self.proj_W))
        # img_gaussian =cv2.GaussianBlur(range_image,(3,3),0)
        img_gaussian = range_image
        # prewitt
        kernelx = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
        img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
        self.partial_r_theta = img_prewitty / (np.pi * 2.0 / self.proj_W) / 6
        self.partial_r_phi = img_prewittx / (((self.fov_up - self.fov_down) / 180.0 * np.pi) / self.proj_H) / 6

        partial_vector = [1.0 * one_matrix, self.partial_r_theta / (range_image * np.cos(self.phi_channel)),
                          self.partial_r_phi / range_image]
        partial_vector = np.asarray(partial_vector)
        partial_vector = np.transpose(partial_vector, (1, 2, 0))
        partial_vector = np.reshape(partial_vector, [self.proj_H, self.proj_W, 3, 1])
        normal_vector = np.matmul(self.R_theta_phi, partial_vector)
        normal_vector = np.squeeze(normal_vector)
        normal_vector = normal_vector / np.reshape(np.max(np.abs(normal_vector), axis=2),
                                                   (self.proj_H, self.proj_W, 1))
        normal_vector_camera = np.zeros((self.proj_H, self.proj_W, 3))
        normal_vector_camera[:, :, 0] = normal_vector[:, :, 1]
        normal_vector_camera[:, :, 1] = -normal_vector[:, :, 2]
        normal_vector_camera[:, :, 2] = normal_vector[:, :, 0]
        return normal_vector_camera

class SemLaserScan(LaserScan):
    """Class that contains LaserScan with x,y,z,r,sem_label,sem_color_label,inst_label,inst_color_label"""
    EXTENSIONS_LABEL = ['.label']

    def __init__(self, sem_color_dict=None, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0, max_classes=300,DA=False,flip_sign=False,rot=False,drop_points=False):
        super(SemLaserScan, self).__init__(project, H, W, fov_up, fov_down,DA=DA,flip_sign=flip_sign,rot=rot,drop_points=drop_points)
        self.reset()

        # make semantic colors
        if sem_color_dict:
            # if I have a dict, make it
            max_sem_key = 0
            for key, data in sem_color_dict.items():
                if key + 1 > max_sem_key:
                    max_sem_key = key + 1
            self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
            for key, value in sem_color_dict.items():
                self.sem_color_lut[key] = np.array(value, np.float32) / 255.0
        else:
            # otherwise make random
            max_sem_key = max_classes
            self.sem_color_lut = np.random.uniform(low=0.0,
                                                   high=1.0,
                                                   size=(max_sem_key, 3))
            # force zero to a gray-ish color
            self.sem_color_lut[0] = np.full((3), 0.1)

        # make instance colors
        max_inst_id = 100000
        self.inst_color_lut = np.random.uniform(low=0.0,
                                                high=1.0,
                                                size=(max_inst_id, 3))
        # force zero to a gray-ish color
        self.inst_color_lut[0] = np.full((3), 0.1)

    def reset(self):
        """ Reset scan members. """
        super(SemLaserScan, self).reset()

        # semantic labels
        self.sem_label = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: label
        self.sem_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

        # instance labels
        self.inst_label = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: label
        self.inst_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

        # projection color with semantic labels
        self.proj_sem_label = np.zeros((self.proj_H, self.proj_W),
                                       dtype=np.int32)  # [H,W]  label
        self.proj_sem_color = np.zeros((self.proj_H, self.proj_W, 3),
                                       dtype=np.float)  # [H,W,3] color

        # projection color with instance labels
        self.proj_inst_label = np.zeros((self.proj_H, self.proj_W),
                                        dtype=np.int32)  # [H,W]  label
        self.proj_inst_color = np.zeros((self.proj_H, self.proj_W, 3),
                                        dtype=np.float)  # [H,W,3] color

    def open_label(self, filename):
        """ Open raw scan and fill in attributes
        """
        # check filename is string
        if not isinstance(filename, str):
            raise TypeError("Filename should be string type, "
                            "but was {type}".format(type=str(type(filename))))

        # check extension is a laserscan
        if not any(filename.endswith(ext) for ext in self.EXTENSIONS_LABEL):
            raise RuntimeError("Filename extension is not valid label file.")

        # if all goes well, open label
        label = np.fromfile(filename, dtype=np.int32)
        label = label.reshape((-1))

        if self.drop_points is not False:
            label = np.delete(label,self.points_to_drop)
        # set it
        self.set_label(label)

    def set_label(self, label):
        """ Set points for label not from file but from np
        """
        # check label makes sense
        if not isinstance(label, np.ndarray):
            raise TypeError("Label should be numpy array")

        # only fill in attribute if the right size
        if label.shape[0] == self.points.shape[0]:
            self.sem_label = label & 0xFFFF  # semantic label in lower half
            self.inst_label = label >> 16  # instance id in upper half
        else:
            print("Points shape: ", self.points.shape)
            print("Label shape: ", label.shape)
            raise ValueError("Scan and Label don't contain same number of points")

        # sanity check
        assert ((self.sem_label + (self.inst_label << 16) == label).all())

        if self.project:
            self.do_label_projection()

    def colorize(self):
        """ Colorize pointcloud with the color of each semantic label
        """
        self.sem_label_color = self.sem_color_lut[self.sem_label]
        self.sem_label_color = self.sem_label_color.reshape((-1, 3))

        self.inst_label_color = self.inst_color_lut[self.inst_label]
        self.inst_label_color = self.inst_label_color.reshape((-1, 3))

    def do_label_projection(self):
        # only map colors to labels that exist
        mask = self.proj_idx >= 0

        # semantics
        self.proj_sem_label[mask] = self.sem_label[self.proj_idx[mask]]
        self.proj_sem_color[mask] = self.sem_color_lut[self.sem_label[self.proj_idx[mask]]]

        # instances
        self.proj_inst_label[mask] = self.inst_label[self.proj_idx[mask]]
        self.proj_inst_color[mask] = self.inst_color_lut[self.inst_label[self.proj_idx[mask]]]
