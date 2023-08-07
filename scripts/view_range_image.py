#!/usr/bin/env python

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()

def range_image_callback(data):
    try:
        # Convert the ROS Image message to a numpy array (OpenCV format)
        cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="mono16")

        # Access a specific pixel (e.g., at row 50, column 100)
        pixel_value = cv_image[:, :]
        rospy.loginfo(f"Pixel (50, 100) range value: {pixel_value.shape}")

        # If you want to access all pixel values:
        # (Be careful with this! It might print too much information)
        # print(cv_image)

    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

def range_image_callback2(data):
    # Here you can access the data in the message
    # For instance, to print the height and width of the image:

    rospy.loginfo(f"sub data: {data.data}")
    #rospy.loginfo(f"Range Image Width: {data.width}, Height: {data.height}")

    # If you need to process the image data, you can access it as:
    # data.data

def listener():
    rospy.init_node('range_image_listener', anonymous=True)
    rospy.Subscriber("/ouster/range_image", Image, range_image_callback)
    
    # Keep the node running
    rospy.spin()

if __name__ == '__main__':
    listener()