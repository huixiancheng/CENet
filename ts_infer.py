import yaml
import os
import torch
import time
import numpy as np
import torch.nn as nn
import shutil

root = 'xxx' # dataset
model_path = 'xxx'   # pretrain_model
result_path = 'xxx'
split = 'test'

ARCH = yaml.safe_load(open(model_path + "/arch_cfg.yaml", 'r'))
DATA = yaml.safe_load(open(model_path + "/data_cfg.yaml", 'r'))


if os.path.isdir(result_path):
    shutil.rmtree(result_path)
os.makedirs(result_path)
for seq in DATA["split"]["valid"]:
    seq = '{0:02d}'.format(int(seq))
    print("valid", seq)
    os.makedirs(os.path.join(result_path, "sequences", seq))
    os.makedirs(os.path.join(result_path, "sequences", seq, "predictions"))
for seq in DATA["split"]["test"]:
    seq = '{0:02d}'.format(int(seq))
    print("test", seq)
    os.makedirs(os.path.join(result_path, "sequences", seq))
    os.makedirs(os.path.join(result_path, "sequences", seq, "predictions"))

with torch.no_grad():
    if ARCH["train"]["pipeline"] == "res":
        from modules.network.ResNet import ResNet_34
        model = ResNet_34(20, ARCH["train"]["aux_loss"])
        def convert_relu_to_softplus(model, act):
            for child_name, child in model.named_children():
                if isinstance(child, nn.LeakyReLU):
                    setattr(model, child_name, act)
                else:
                    convert_relu_to_softplus(child, act)
        if ARCH["train"]["act"] == "Hardswish":
            convert_relu_to_softplus(model, nn.Hardswish())
        elif ARCH["train"]["act"] == "SiLU":
            convert_relu_to_softplus(model, nn.SiLU())
# print(model)

w_dict = torch.load(model_path + "/SENet_valid_best",
                    map_location=lambda storage, loc: storage)
model.load_state_dict(w_dict['state_dict'], strict=True)

post = None

from postproc.KNN import KNN

if ARCH["post"]["KNN"]["use"]:
    post = KNN(ARCH["post"]["KNN"]["params"], 20)

import torch.backends.cudnn as cudnn
gpu = False
model_single = model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Infering in device: ", device)
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
  cudnn.benchmark = True
  cudnn.fastest = True
  gpu = True
  model.cuda()

if split == 'valid':
    all_seq_list = ['08']
else:
    all_seq_list = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']

from common.laserscan import LaserScan

sensor = ARCH["dataset"]["sensor"]
max_points = ARCH["dataset"]["max_points"]
sensor_img_means = torch.tensor(sensor["img_means"], dtype=torch.float)
sensor_img_stds = torch.tensor(sensor["img_stds"], dtype=torch.float)

cnn =[]
knn = []

def map(label, mapdict):
    # put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # make learning map a lookup table
    maxkey = 0
    for key, data in mapdict.items():
        if isinstance(data, list):
            nel = len(data)
        else:
            nel = 1
        if key > maxkey:
            maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
        lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
        lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in mapdict.items():
        try:
            lut[key] = data
        except IndexError:
            print("Wrong key ", key)
    # do the mapping
    return lut[label]

def to_original(label):
    # put label in original values
    return map(label, DATA["learning_map_inv"])


model.eval()


for seq_name in all_seq_list:
    import glob
    lidar_list = glob.glob(root + '/sequences/' + seq_name + '/velodyne/*.bin')
    lidar_list.sort()

    for i in range(len(lidar_list)):
        scan = LaserScan(project=True, H=sensor["img_prop"]["height"], W=sensor["img_prop"]["width"],
                       fov_up=sensor["fov_up"], fov_down=sensor["fov_down"],
                       DA=False, flip_sign=False, rot=False, drop_points=False)

        scan.open_scan(lidar_list[i])
        unproj_n_points = scan.points.shape[0]
        unproj_xyz = torch.full((max_points, 3), -1.0, dtype=torch.float)
        unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
        unproj_range = torch.full([max_points], -1.0, dtype=torch.float)
        unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
        unproj_remissions = torch.full([max_points], -1.0, dtype=torch.float)
        unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)
        unproj_labels = []

        # get points and labels
        proj_range = torch.from_numpy(scan.proj_range).clone()
        proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
        proj_remission = torch.from_numpy(scan.proj_remission).clone()

        proj_mask = torch.from_numpy(scan.proj_mask)
        proj_labels = []

        proj_x = torch.full([max_points], -1, dtype=torch.long)
        proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
        proj_y = torch.full([max_points], -1, dtype=torch.long)
        proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)

        proj = torch.cat([proj_range.unsqueeze(0).clone(),
                          proj_xyz.clone().permute(2, 0, 1),
                          proj_remission.unsqueeze(0).clone()])

        proj = (proj - sensor_img_means[:, None, None]
                ) / sensor_img_stds[:, None, None]

        proj = proj * proj_mask.float()

        # get name and sequence
        path_norm = os.path.normpath(lidar_list[i])
        path_split = path_norm.split(os.sep)
        path_seq = path_split[-3]
        path_name = path_split[-1].replace(".bin", ".label")

        npoints = torch.tensor(unproj_n_points).unsqueeze(dim=0)
        p_x = proj_x.unsqueeze(dim=0)
        p_y = proj_y.unsqueeze(dim=0)
        proj_in = proj.unsqueeze(dim=0)
        proj_range = proj_range.unsqueeze(dim=0)
        unproj_range = unproj_range.unsqueeze(dim=0)

        p_x = p_x[0, :npoints]
        p_y = p_y[0, :npoints]
        proj_range = proj_range[0, :npoints]
        unproj_range = unproj_range[0, :npoints]
        path_seq = path_seq
        path_name = path_name

        proj_in = proj_in.cuda()
        p_x = p_x.cuda()
        p_y = p_y.cuda()
        if post:
            proj_range = proj_range.cuda()
            unproj_range = unproj_range.cuda()

        end = time.time()
        if ARCH["train"]["aux_loss"]:
            with torch.cuda.amp.autocast(enabled=True):
                [proj_output, x_2, x_3, x_4] = model(proj_in)
        else:
            with torch.cuda.amp.autocast(enabled=True):
                proj_output = model(proj_in)

        proj_argmax = proj_output[0].argmax(dim=0)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        res = time.time() - end
        print("Network seq", path_seq, "scan", path_name,
              "in", res, "sec")
        end = time.time()
        cnn.append(res)

        if post:
            # knn postproc
            unproj_argmax = post(proj_range,
                                      unproj_range,
                                      proj_argmax,
                                      p_x,
                                      p_y)
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
        pred_np = unproj_argmax.cpu().numpy()
        pred_np = pred_np.reshape((-1)).astype(np.int32)

        # map to original label
        pred_np = to_original(pred_np)

        # save scan
        path = os.path.join(result_path, "sequences",
                            path_seq, "predictions", path_name)
        pred_np.tofile(path)

    print("Mean CNN inference time:{}\t std:{}".format(np.mean(cnn), np.std(cnn)))
    print("Mean KNN inference time:{}\t std:{}".format(np.mean(knn), np.std(knn)))
    print("Total Frames:{}".format(len(cnn)))
    print("Finished Infering")