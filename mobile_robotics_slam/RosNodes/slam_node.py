import rclpy
import os
import sys
import signal
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry


path = __file__
file_location_subfolders = 3 #Number of folder to go up to reach root of package
for _ in range(file_location_subfolders):
    path = os.path.dirname(path)
sys.path.insert(0, path)

from mobile_robotics_slam.Keypoint import  KeypointList

NN_MATCHING_THRESHOLD = 0.05
NN_NEW_LANDMARK_THRESHOLD = 0.2

SAVE_KEYPOINTS = True


class SlamNode(Node):

##################################################################
# Initialization
##################################################################
    def __init__(self):
        super().__init__("slam_node", parameter_overrides=[]) 
        # Declare variables
        self.robot_real_x = 0.0
        self.robot_real_y = 0.0
        self.robot_real_phi = 0.0
        self.falko_keypoints = KeypointList()
        self.oc_keypoints = KeypointList()
        self.my_extractor_keypoints = KeypointList()
        
        self.scan_subscription = self.create_subscription(LaserScan, "/diff_drive/scan", self.scan_callback, 10)
        self.filtered_scan_publisher = self.create_publisher(LaserScan, "/diff_drive/filtered_scan", 10)
        self.keypoints_publisher = self.create_publisher(MarkerArray, "/diff_drive/keypoints", 10)

        ## Create a common callback for both the keypoints that takes as argument the keypoint list
        self.pose_subscription = self.create_subscription(Odometry, "/diff_drive/real_pose",  self.pose_callback, 10) 
        self.FALKO_Keypoints_subscriber = self.create_subscription(MarkerArray, "/falko_keypoints", lambda msg :self.KeyPoints_callback(msg, self.falko_keypoints, "FALKO"), 10) # type: ignore
        self.OC_KeyPoints_subscriber = self.create_subscription(MarkerArray, "/oc_keypoints", lambda msg : self.KeyPoints_callback(msg, self.oc_keypoints, "OC"), 10) # type: ignore
        self.my_extractor_keypoints_subscriber = self.create_subscription(MarkerArray, "/corner_keypoints", lambda msg : self.KeyPoints_callback(msg, self.my_extractor_keypoints, "My Extractor"), 10) # type: ignore

##################################################################
# Function To Modify When Adding New Extractor
##################################################################

    def signal_handler (self, sig, frame):
        if SAVE_KEYPOINTS:
            # For Each Keypoint List, Get the infos
            falko_keypoints = self.falko_keypoints.get_keypoint_info()
            oc_keypoints = self.oc_keypoints.get_keypoint_info()
            my_keypoints = self.my_extractor_keypoints.get_keypoint_info()
            
            # For Each Keypoint List, Save the keypoints to a file
            with open(os.path.join(path, "example_scans", "falko_keypoints.txt"), "w") as f:
                for keypoint in falko_keypoints:
                    f.write(" ".join([str(k) for k in keypoint]) + "\n")

            with open(os.path.join(path, "example_scans", "oc_keypoints.txt"), "w") as f:
                for keypoint in oc_keypoints:
                    f.write(" ".join([str(k) for k in keypoint]) + "\n")

            with open(os.path.join(path, "example_scans", "my_keypoints.txt"), "w") as f:
                for keypoint in my_keypoints:
                    f.write(" ".join([str(k) for k in keypoint]) + "\n")

            self.get_logger().info("Saved the keypoints to the file")

        self.destroy_node()
        rclpy.try_shutdown()


    def pose_callback(self, msg: Odometry):
        """Callback function that updates the robot pose and then publishes the keypoints as markers in the lidar frame""" 
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        phi = self.quaternion_to_euler(msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w,)
        self.set_robot_real_pose(x, y, phi)

        markers = MarkerArray()

        #Transform the keypoints to the real pose frame for each keypoint list
        #falko_lidar_frame = self.transform_keypoints_to_lidar_frame(self.falko_keypoints)
        falko_lidar_frame = self.transform_stable_keypoints_to_lidar_frame(self.falko_keypoints)
        falko_markers = self.generate_markers_from_keypoints(falko_lidar_frame, color=[1.0, 0.0, 0.0], base_idx=0)

        #oc_lidar_frame = self.transform_keypoints_to_lidar_frame(self.oc_keypoints)
        oc_lidar_frame = self.transform_stable_keypoints_to_lidar_frame(self.oc_keypoints)
        oc_markers = self.generate_markers_from_keypoints(oc_lidar_frame, color=[0.0, 1.0, 0.0], base_idx=len(falko_lidar_frame))

        #my_extractor_lidar_frame = self.transform_keypoints_to_lidar_frame(self.my_extractor_keypoints)
        my_extractor_lidar_frame = self.transform_stable_keypoints_to_lidar_frame(self.my_extractor_keypoints)
        my_extractor_markers = self.generate_markers_from_keypoints(my_extractor_lidar_frame, color=[0.0, 0.0, 1.0], base_idx=len(falko_lidar_frame) + len(oc_lidar_frame))

        markers.markers.extend(falko_markers.markers) # type: ignore
        markers.markers.extend(oc_markers.markers) # type: ignore
        markers.markers.extend(my_extractor_markers.markers) # type: ignore
        
        self.keypoints_publisher.publish(markers)

    def scan_callback(self, msg: LaserScan):
        pass

##################################################################
# Common Functions for All Extractors
##################################################################

    def KeyPoints_callback(self, msg: MarkerArray, keypoint_list: KeypointList, extractor_name: str):
        """Callback function for the keypoints that updates the matched keypoints and adds the new keypoints to the list"""
        keypoints = []
        for marker in msg.markers:
            keypoints.append([marker.pose.position.x, marker.pose.position.y])
        match, non_match_idx, ambiguous_idx = keypoint_list.match_keypoints_NN(keypoints, NN_MATCHING_THRESHOLD, NN_NEW_LANDMARK_THRESHOLD)
        for match in match:
            keypoint_list.keypoints[match[1]].update(keypoints[match[0]][0], keypoints[match[0]][1])
        for idx in non_match_idx:
            keypoint_list.add_keypoint(keypoints[idx][0], keypoints[idx][1], keypoint_list.number_of_received_set)
        self._logger.info(f"{extractor_name} Total number of Stored keypoints: {len(keypoint_list.keypoints)}")
        self._logger.info(f"{extractor_name} Total number of Stable keypoints: {len(keypoint_list.get_stable_keypoints())}")

    def generate_markers_from_keypoints(self, keypoints, color, base_idx):
        """Generates markers from the keypoints given the color and the base index"""
        markers = MarkerArray()
        for idx, keypoint in enumerate(keypoints):
            marker = Marker()
            marker.header.frame_id = "lidar_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = idx + base_idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(keypoint[0])
            marker.pose.position.y = float(keypoint[1])
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.lifetime.nanosec = int(1e+8)
            markers.markers.append(marker) # type: ignore

        return markers

    def transform_keypoints_to_lidar_frame(self, keypoint_list: KeypointList):
        """Transforms the keypoints to the real pose frame"""
        keypoints_lidar_frame = []
        if len(keypoint_list.keypoints) > 0:
            keypoints = keypoint_list.get_keypoints()
            keypoints_lidar_frame = [self.real_pose_frame_to_lidar_frame(keypoint[0], keypoint[1]) for keypoint in keypoints]
            return keypoints_lidar_frame
        return keypoints_lidar_frame
    
    def transform_stable_keypoints_to_lidar_frame(self, keypoint_list: KeypointList):
        """Transforms the stable keypoints to the real pose frame"""
        stable_keypoints_lidar_frame = []
        if len(keypoint_list.keypoints) > 0:
            stable_keypoints = keypoint_list.get_stable_keypoints()
            stable_keypoints_lidar_frame = [self.real_pose_frame_to_lidar_frame(keypoint[0], keypoint[1]) for keypoint in stable_keypoints]
            return stable_keypoints_lidar_frame
        return stable_keypoints_lidar_frame
        
    def lidar_frame_to_real_pose_frame(self, x, y):
            """Transforms the lidar frame to the real pose frame"""
            x_real = self.robot_real_x + x * np.cos(self.robot_real_phi) - y * np.sin(self.robot_real_phi)
            y_real = self.robot_real_y + x * np.sin(self.robot_real_phi) + y * np.cos(self.robot_real_phi)
            return [x_real, y_real]

    def real_pose_frame_to_lidar_frame(self, x, y):
        """Transforms the real pose frame to the lidar frame"""
        x_lidar = (x - self.robot_real_x) * np.cos(self.robot_real_phi) + (y - self.robot_real_y) * np.sin(self.robot_real_phi)
        y_lidar = -(x - self.robot_real_x) * np.sin(self.robot_real_phi) + (y - self.robot_real_y) * np.cos(self.robot_real_phi)
        return [x_lidar, y_lidar]

    def set_robot_real_pose(self, pose_x, pose_y, phi):
        """Set the real robot pose"""
        self.robot_real_x = pose_x
        self.robot_real_y = pose_y
        self.robot_real_phi = phi

    def quaternion_to_euler(self, x, y, z, w):
        """Converts quaternion to euler angles"""
        phi = np.arctan2(2.0 * (w * z + x * y), w * w + x * x - y * y - z * z)
        return phi
    
    def euler_to_quaternion(self, phi):
        """Converts euler angles to quaternion"""
        x = 0.0
        y = 0.0
        z = np.sin(phi / 2)
        w = np.cos(phi / 2)
        return x, y, z, w
    
    def polar_to_cartesian(self, range, angle):
        """Converts polar coordinates to cartesian
        Input:  range: distance to the object
                angle: angle to the object
                Output: array with x and y coordinates"""
        x = range * np.cos(angle)
        y = range * np.sin(angle)
        return np.array([x, y])
    
    def cartesian_to_polar(self, x, y):
        """Converts cartesian coordinates to polar
        Input:  x: x coordinate
                y: y coordinate
                Output: array with range and angle"""
        range = np.sqrt(x**2 + y**2)
        angle = np.arctan2(y, x)
        return np.array([range, angle])
        
    
    

def main(args=None):
    rclpy.init(args=args)

    print("SLAM node started")

    slam_node = SlamNode()
    signal.signal(signal.SIGINT, slam_node.signal_handler )

    try:
        rclpy.spin(slam_node)
    except KeyboardInterrupt:
        pass
    finally:
        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        slam_node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
