#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

#import argparse
#import subprocess
#import datetime
import yaml
from shutil import copyfile
import os
import shutil
from modules.user_ros import User
import rospkg

if __name__ == '__main__':
    splits = ["train", "valid", "test"]

    # Get the path of the ROS package
    rospack = rospkg.RosPack()
    ce_net_ros_path = rospack.get_path('ce_net')
    hunter_robot_data_path = rospack.get_path('hunter_robot_data')

    model_dir = f'{ce_net_ros_path}/src/dataset/final_result/1024+valid5'
    log_dir = f'{ce_net_ros_path}/src/predictions'
    dataset_dir = f'{hunter_robot_data_path}/bin/cu_campus'
    split = '13'

    # print summary of what we will do
    print("----------")
    print("INTERFACE:")
    print("dataset", dataset_dir)
    print("log", log_dir)
    print("model", model_dir)
    print("infering", split)
    print("----------\n")

    # open arch config file
    try:
        print("Opening arch config file from %s" % model_dir)
        ARCH = yaml.safe_load(open(model_dir + "/arch_cfg.yaml", 'r'))
    except Exception as e:
        print(e)
        print("Error opening arch yaml file.")
        quit()

    # open data config file
    try:
        print("Opening data config file from %s" % model_dir)
        DATA = yaml.safe_load(open(model_dir + "/data_cfg.yaml", 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

    # create log folder
    '''
    try:
        if os.path.isdir(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir)
        os.makedirs(os.path.join(log_dir, "sequences"))
        for seq in DATA["split"]["test"]:
            seq = '{0:02d}'.format(int(seq))
            print("test", seq)
            os.makedirs(os.path.join(log_dir, "sequences", seq))
            os.makedirs(os.path.join(log_dir, "sequences", seq, "predictions"))
    except Exception as e:
        print(e)
        print("Error creating log directory. Check permissions!")
        raise
    '''
    # does model folder exist?
    if os.path.isdir(model_dir):
        print("model folder exists! Using model from %s" % (model_dir))
    else:
        print("model folder doesnt exist! Can't infer...")
        quit()

    # create user and infer dataset
    user = User(ARCH, DATA, dataset_dir, log_dir, model_dir, split)
    user.listener()
    
    #user.infer()
