#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import numpy as np
import ros_numpy
import struct
from modules.user import User

def callback(data):
    rospy.loginfo('Received a PointCloud2 message')
    global pc_number

    # Convert the point cloud data to a np array
    #points = np.array(list(point_cloud2.read_points(data, field_names=("x", "y", "z", "intensity", "t", "ring"))))
    #print(points)
    #rospy.loginfo('Fields in the PointCloud2 message: %s', data.fields)
    
    point_type = np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('intensity', np.float32),
        ('t', np.uint32),
        ('ring', np.uint8)
    ])

    pc = ros_numpy.numpify(data)

    data_points=np.zeros((pc.shape[0], pc.shape[1], 4))
    data_points[:,:,0]=pc['x']
    data_points[:,:,1]=pc['y']
    data_points[:,:,2]=pc['z']
    data_points[:,:,3]=pc['intensity']

    #points = []

    # Loop through each point in pc and append its attributes to points
    points = np.zeros(pc.shape, dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32), 
                                    ('intensity', np.float32), ('t', np.uint32), ('ring', np.uint8)])

    points['x'] = pc['x'].astype(np.float32)
    points['y'] = pc['y'].astype(np.float32)
    points['z'] = pc['z'].astype(np.float32)
    points['intensity'] = pc['intensity'].astype(np.float32)
    points['t'] = pc['t'].astype(np.uint32)
    points['ring'] = pc['ring'].astype(np.uint8)
    
    # Convert points to a numpy array
    points = np.array(points, dtype=point_type)
    print(f"points: {points}")

    z_buff = 10
    pc_num_str = str(pc_number).zfill(z_buff)

    # Save files
    data_bin_file = f'/home/donceykong/hunter_ws/src/hunter_robot/hunter_robot_data/bin/cu_campus/sequences/13/velodyne/{pc_num_str}.bin'
    data_points.astype('float32').tofile(data_bin_file)

    leo_bin_file = f'/home/donceykong/hunter_ws/src/hunter_robot/hunter_robot_data/bin/cu_campus/sequences/13/leo_points/{pc_num_str}.bin'
    points.astype(point_type).tofile(leo_bin_file)

    timestamp_file = f'/home/donceykong/hunter_ws/src/hunter_robot/hunter_robot_data/bin/cu_campus/sequences/13/leo_points/{pc_num_str}.time'
    #timestamp = np.float64(data.header.stamp.to_sec())
    #timestamp.astype('float64').tofile(timestamp_file)
    with open(timestamp_file, "w") as f:
        f.write(str(data.header.stamp.to_sec()))  # Save the timestamp as seconds since epoch

    #pcd_file = f'/home/donceykong/Desktop/hunter_ws/src/hunter_robot/hunter_robot_data/pcd/07_17_2023/{pc_num_str}.pcd'
    #pcd = bin_to_pcd(bin_file)
    #o3d.io.write_point_cloud(pcd_file, pcd)

    pc_number += 1

def listener():
    global pub
    global pc_number

    # Init pc number
    pc_number = 0
    
    rospy.init_node('pointcloud_to_bin_file', anonymous=True)

    # Subscribe to the input PointCloud2 topic
    sc_lio_sam_global_map = '/sc_lio_sam/map_global'
    sc_lio_sam_local_map = '/sc_lio_sam/map_local'
    ouster_points = '/ouster/points'

    rospy.Subscriber(ouster_points, PointCloud2, callback)

    # Create a publisher for the output PointCloud2 topic
    #pub = rospy.Publisher('/output_cloud', PointCloud2, queue_size=10)

    rospy.spin()

if __name__ == '__main__':
    listener()