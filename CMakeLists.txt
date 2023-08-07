cmake_minimum_required(VERSION 2.8.3)
project(ce_net)

find_package(catkin REQUIRED)

find_package(catkin REQUIRED COMPONENTS
    rospy
)

catkin_package()
catkin_python_setup()

catkin_install_python(PROGRAMS scripts/infer.py
                      scripts/infer_ros.py
                      scripts/view_range_image.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
