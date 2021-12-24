#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import time

import numpy as np
import math
import random
from scipy.spatial.transform import Rotation as R

class LaserScan:
    """Class that contains LaserScan with x,y,z,r"""
    EXTENSIONS_SCAN = ['.bin']

    def __init__(self, project=False, H=40, W=1800, fov_up=7.0, fov_down=-16.0,DA=False,flip_sign=False,rot=False,drop_points=False):
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
        self.tags = np.full((self.proj_H*self.proj_W), False, dtype=np.bool)

        self.unproj_range = np.zeros((0, 1), dtype=np.float32)

#         self.midrange = np.zeros((self.proj_H * self.proj_W), dtype=np.uint8)
#         self.midremission = np.zeros((self.proj_H * self.proj_W), dtype=np.float32)
#         self.midxyz = np.zeros((self.proj_H * self.proj_W, 3), dtype=np.float32)
        
        self.midrange = np.full((self.proj_H * self.proj_W), -1, dtype=np.float32)
        self.midremission = np.full((self.proj_H * self.proj_W), -1, dtype=np.float32)
        self.midxyz = np.full((self.proj_H * self.proj_W, 3), -1, dtype=np.float32)

        # projected range image - [H,W] range (-1 is no data)
        self.proj_range = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)
        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1, dtype=np.float32)
        # projected remission - [H,W] intensity (-1 is no data)
        self.proj_remission = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)


    def size(self):
        """ Return the size of the point cloud. """
        return self.points.shape[0]

    def __len__(self):
        return self.size()

    def open_scan(self, filename, tagname):
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

        tags = np.fromfile(tagname, dtype=np.bool)

#         if self.rot:
#             euler_angle = np.random.normal(-45, 45, 1)[0]  # 40
#             r = np.array(R.from_euler('zyx', [[euler_angle, 0, 0]], degrees=True).as_matrix())
#             r_t = r.transpose()
#             points = points.dot(r_t)
#             points = np.squeeze(points)

        if self.DA:
            shift_x = np.random.normal(0.0, 0.7, 1)[0]
            shift_y = np.random.normal(0.0, 0.7, 1)[0]
            shift_z = np.random.normal(0.0, 0.007, 1)[0]
            points[:, 0] = points[:, 0] + shift_x
            points[:, 1] = points[:, 1] + shift_y
            points[:, 2] = points[:, 2] + shift_z

#         if self.flip_sign:
#             if random.random() > 0.5:
#                 points[:, 0] = -points[:, 0]
#             if random.random() > 0.5:
#                 points[:, 1] = -points[:, 1]

        self.set_points(points, remissions, tags)

    def set_points(self, points, remissions=None, tags=None):
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
        self.points = points  # get xyz
        self.tags = tags
        if remissions is not None:
            self.remissions = remissions  # get remission
        else:
            self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

        # if projection is wanted, then do it and fill in the structure
        if self.project:
            self.do_range_projection()


    def do_range_projection(self):
        """ Project a pointcloud into a spherical projection image.projection.
            Function takes no arguments because it can be also called externally
            if the value of the constructor was not set (in case you change your
            mind about wanting the projection)
        """
        # laser parameters

#         dis = np.linalg.norm(self.points, axis=1) * 5
#         dis = np.minimum(dis, 255)
#         dis = dis.astype(np.uint8)

        dis = np.linalg.norm(self.points, 2, axis=1)
    
        self.midremission[self.tags] = self.remissions
        self.midrange[self.tags] = dis
        self.midxyz[self.tags] = self.points

        self.unproj_range = np.copy(dis)

        self.proj_remission = np.reshape(self.midremission, (self.proj_H, self.proj_W))
        self.proj_range = np.reshape(self.midrange, (self.proj_H, self.proj_W))
        self.proj_xyz = np.reshape(self.midxyz, (self.proj_H, self.proj_W, 3))
        # get depth of all points



class SemLaserScan(LaserScan):
    """Class that contains LaserScan with x,y,z,r,sem_label,sem_color_label,inst_label,inst_color_label"""
    EXTENSIONS_LABEL = ['.label']

    def __init__(self, sem_color_dict=None, project=False, H=40, W=1800, fov_up=7.0, fov_down=-16.0, max_classes=300,DA=False,flip_sign=False,rot=False,drop_points=False):
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


    def reset(self):
        """ Reset scan members. """
        super(SemLaserScan, self).reset()

        # semantic labels
        self.sem_label = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: label
        self.sem_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

        self.midsemlabel = np.zeros((self.proj_H * self.proj_W), dtype=np.int32)

        # projection color with semantic labels
        self.proj_sem_label = np.zeros((self.proj_H, self.proj_W),
                                       dtype=np.int32)  # [H,W]  label
        self.proj_sem_color = np.zeros((self.proj_H, self.proj_W, 3),
                                       dtype=np.float)  # [H,W,3] color

    def open_label(self, filename, tagname):
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
        tags = np.fromfile(tagname, dtype=np.bool)

        # if self.drop_points is not False:
        #     label = np.delete(label,self.points_to_drop)
        # set it
        self.set_label(label, tags)

    def set_label(self, label, tags):
        """ Set points for label not from file but from np
        """
        # check label makes sense
        if not isinstance(label, np.ndarray):
            raise TypeError("Label should be numpy array")

        # only fill in attribute if the right size
        if label.shape[0] == self.points.shape[0]:
            self.sem_label = label & 0xFFFF  # semantic label in lower half
        else:
            print("Points shape: ", self.points.shape)
            print("Label shape: ", label.shape)
            raise ValueError("Scan and Label don't contain same number of points")
        self.tags = tags

        if self.project:
            self.do_label_projection()

    def colorize(self):
        """ Colorize pointcloud with the color of each semantic label
        """
        self.sem_label_color = self.sem_color_lut[self.sem_label]
        self.sem_label_color = self.sem_label_color.reshape((-1, 3))

    def do_label_projection(self):
        # only map colors to labels that exist
        # mask = self.proj_idx >= 0

        self.midsemlabel[self.tags] = self.sem_label
        self.proj_sem_label = np.reshape(self.midsemlabel, (self.proj_H, self.proj_W))

        self.proj_sem_color = self.sem_color_lut[self.proj_sem_label]

        # semantics
        # self.proj_sem_label[mask] = self.sem_label[self.proj_idx[mask]]
        # self.proj_sem_color[mask] = self.sem_color_lut[self.sem_label[self.proj_idx[mask]]]


