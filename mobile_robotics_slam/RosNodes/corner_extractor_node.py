import rclpy
import os
import sys
import signal
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Pose


# adding localization_lib to the system path
sys.path.insert(
    0, os.path.join(os.getcwd(), "src", "application", "mobile_robotics_slam")
)


from mobile_robotics_slam.Extractors.Corners.CornerExtractor import CornerExtractor

class MyCornerExtractor(Node):

##################################################################
# Initialization
##################################################################
    def __init__(self):
        super().__init__("slam_node", parameter_overrides=[]) 
        

        # Declare variables
        self.robot_real_x = 0.0
        self.robot_real_y = 0.0
        self.robot_real_phi = 0.0
        self.extractor = CornerExtractor()
        self.setup_extractor_parameters()

        self.keypoints_publisher = self.create_publisher(MarkerArray, "/corner_keypoints", 10)
        self.corner_poses_publisher = self.create_publisher(PoseArray, "/corner_poses", 10)
        self.scan_subscription = self.create_subscription(LaserScan, "/diff_drive/scan", self.scan_callback, 10)
        
        self.pose_subscription = self.create_subscription(Odometry, "/diff_drive/real_pose",  self.pose_callback, 10) 
        #self.pose_subscription = self.create_subscription(Odometry, "/dingo/odometry",  self.pose_callback, 10)

    def setup_extractor_parameters(self):
        # Set the parameters of the Corner Extractor
        min_corner_angle = 70
        max_corner_angle = 110
        max_intersecton_distance = 0.8
        self.extractor.set_corner_params(max_intersecton_distance, min_corner_angle, max_corner_angle)

        # Set the parameters of the Adaptive Segment Detector
        sigma_ranges = 0.15
        lambda_angle = 10
        merge_distance = 0.07
        min_points_density = 2
        min_segment_length = 0.5
        self.extractor.set_detector_params(sigma_ranges, lambda_angle, merge_distance, min_points_density, min_segment_length)

        # Set the parameters of the Segment Handler
        epsilon = 0.1
        min_density_after_segmentation = 10
        min_length_after_segmentation = 0.9
        self.extractor.set_handler_params(epsilon, min_density_after_segmentation, min_length_after_segmentation)

    

    def pose_callback(self, msg: Odometry):
        """Callback function that updates the robot pose and then publishes the keypoints as markers in the lidar frame""" 
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        phi = self.quaternion_to_euler(msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w,)
        self.set_robot_real_pose(x, y, phi)


    def scan_callback(self, msg: LaserScan):
        """Callback function that processes the scan and publishes the keypoints as markers in the lidar frame"""
        ranges = np.array(msg.ranges)
        field_of_view = msg.angle_max - msg.angle_min
        angle_min = msg.angle_min

        # Store Timestamp
        scan_time = msg.header.stamp
        # Get the real pose of the robot before processing the scan
        robot_pose = np.array([self.robot_real_x, self.robot_real_y, self.robot_real_phi])
        # Extract the keypoints
        self.extractor.extract_corners(ranges, field_of_view, angle_min, robot_pose)
        corners = self.extractor.get_corners()
        # Transform the keypoints to the real pose frame
        #corners_real_pose = [self.lidar_frame_to_pose_frame(corner[0], corner[1], robot_pose) for corner in corners]
        corners_real_pose = corners
        for corner in corners_real_pose:
            print("Corner: ", corner.x, corner.y, corner.orientation)
        # Publish the keypoints as markers
        keypoints_markers = self.generate_markers_from_keypoints(corners_real_pose, [1.0, 0.0, 0.0], 0, scan_time)
        self.generate_poses_from_keypoint(corners, scan_time)
        self.keypoints_publisher.publish(keypoints_markers)
        self.get_logger().info("Extracted keypoints: " + str(len(corners)))

    def generate_poses_from_keypoint(self, keypoints, timestamp):
        """Generates poses from the keypoints given the color and the base index"""
        
        poses = PoseArray()
        poses.header.frame_id = "lidar_link"
        poses.header.stamp = timestamp
        for idx, keypoint in enumerate(keypoints):
            pose = Pose()
            pose.position.x = float(keypoint.x)
            pose.position.y = float(keypoint.y)
            pose.position.z = 0.0
            pose.orientation.x = keypoint.orientation_quaternion[0]
            pose.orientation.y = keypoint.orientation_quaternion[1]
            pose.orientation.z = keypoint.orientation_quaternion[2]
            pose.orientation.w = keypoint.orientation_quaternion[3]
            poses.poses.append(pose) # type: ignore

        self.corner_poses_publisher.publish(poses)
        

        

    def generate_markers_from_keypoints(self, keypoints, color, base_idx, timestamp):
        """Generates markers from the keypoints given the color and the base index"""
        
        markers = MarkerArray()
        for idx, keypoint in enumerate(keypoints):
            marker = Marker()
            marker.header.frame_id = "lidar_link"
            marker.header.stamp = timestamp
            marker.id = idx + base_idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(keypoint.x)
            marker.pose.position.y = float(keypoint.y)
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.7
            marker.color.a = 1.0
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.lifetime.nanosec = int(1e8)

            markers.markers.append(marker) # type: ignore

        return markers

        
    def lidar_frame_to_pose_frame(self, x, y, pose):
            """Transforms the lidar frame to the real pose frame"""
            robot_x = pose[0]
            robot_y = pose[1]
            robot_phi = pose[2]
            x_real = robot_x + x * np.cos(robot_phi) - y * np.sin(robot_phi)
            y_real = robot_y + x * np.sin(robot_phi) + y * np.cos(robot_phi)
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

    slam_node = MyCornerExtractor()
    # signal.signal(signal.SIGINT, slam_node.signal_handler )

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
