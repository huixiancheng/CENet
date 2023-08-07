#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import Image
import cv2

def range_image_callback(data):
    # Ensure the encoding is as expected
    if data.encoding != "mono16":
        rospy.logerr(f"Unexpected encoding: {data.encoding}")
        return
    
    # Convert the byte data to a numpy array
    dtype = np.dtype(np.uint16)  # as it's mono16
    dtype = dtype.newbyteorder('>')  # ROS Image messages use big endian
    cv_image = np.frombuffer(data.data, dtype=dtype).reshape(data.height, data.width)
    
    # Access a specific pixel (e.g., at row 50, column 100)
    #pixel_value = cv_image[50, 100]

#    rospy.loginfo(f"Pixel (50, 100) range value: {pixel_value}")

    cv2.imshow("Model image", cv_image)
    
    key = cv2.waitKey(10) & 0xFF
#    cv2.waitKey(0)
    #cv2.destroyAllWindows()

def listener():
    rospy.init_node('range_image_listener', anonymous=True)
    rospy.Subscriber("/ouster/range_image", Image, range_image_callback)

    # Keep the node running
    rospy.spin()

if __name__ == '__main__':
    listener()