cmake_minimum_required(VERSION 3.0.2)
project(mobile_robotics_slam)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  nav_msgs
  sensor_msgs
)



catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES mobile_robotics_slam
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)
catkin_install_python(
  PROGRAMS
    mobile_robotics_slam/RosNodes/corner_extractor_node_ros1.py
    mobile_robotics_slam/RosNodes/graph_slam_node_Real_Reflector.py
    mobile_robotics_slam/RosNodes/graph_slam_node_ros1.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY mobile_robotics_slam/
  DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}/mobile_robotics_slam
  FILES_MATCHING PATTERN "*.py"
)
# Install the package.xml file
install(
  FILES package.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)