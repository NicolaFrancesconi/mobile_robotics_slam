import rospy
import os
import sys
import numpy as np
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Pose

path = __file__
file_location_subfolders = 3  # Number of folders to go up to reach the root of the package
for _ in range(file_location_subfolders):
    path = os.path.dirname(path)
sys.path.insert(0, path)

from mobile_robotics_slam.Extractors.Corners.CornerExtractor import CornerExtractor

def remove_png_files(folder_path):
    try:
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".png"):
                file_path = os.path.join(folder_path, file_name)
                os.remove(file_path)
                print(f"Removed: {file_path}")
    except Exception as e:
        print(f"Error: {e}")


class MyCornerExtractor:

    def __init__(self):
        # Initialize the ROS node
        rospy.init_node("slam_node", anonymous=True)

        # Declare variables
        self.robot_real_x = 0.0
        self.robot_real_y = 0.0
        self.robot_real_phi = 0.0
        self.extractor = CornerExtractor()
        self.setup_extractor_parameters()

        #print("Path To remove png", path)
        remove_png_files(path)

        self.keypoints_publisher = rospy.Publisher("/corner_keypoints", MarkerArray, queue_size=10)
        self.corner_poses_publisher = rospy.Publisher("/corner_poses", PoseArray, queue_size=10)

        rospy.Subscriber("/scan", LaserScan, self.scan_callback)
        rospy.Subscriber("/odometry/filtered", Odometry, self.pose_callback)

    def setup_extractor_parameters(self):
        # Set the parameters of the Corner Extractor
        min_corner_angle = 70
        max_corner_angle = 110
        max_intersecton_distance = 0.3
        self.extractor.set_corner_params(max_intersecton_distance, min_corner_angle, max_corner_angle)

        # Set the parameters of the Adaptive Segment Detector
        sigma_ranges = 0.20
        lambda_angle = 10
        merge_distance = 0.15
        min_points_density = 4
        min_segment_length = 0.1
        self.extractor.set_detector_params(sigma_ranges, lambda_angle, merge_distance, min_points_density, min_segment_length)

        # Set the parameters of the Segment Handler
        epsilon = 0.12
        min_density_after_segmentation = 4
        min_length_after_segmentation = 0.2
        self.extractor.set_handler_params(epsilon, min_density_after_segmentation, min_length_after_segmentation)

    def pose_callback(self, msg):
        """Callback function that updates the robot pose"""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        phi = self.quaternion_to_euler(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )
        self.set_robot_real_pose(x, y, phi)

    def scan_callback(self, msg):
        """Callback function that processes the scan and publishes the keypoints as markers in the lidar frame"""
        ranges = np.array(msg.ranges)
        field_of_view = msg.angle_max - msg.angle_min
        angle_min = msg.angle_min

        # Store Timestamp
        scan_time = msg.header.stamp
        # Get the real pose of the robot before processing the scan
        robot_pose = np.array([self.robot_real_x, self.robot_real_y, self.robot_real_phi])
        # Extract the keypoints
        self.extractor.extract_corners(ranges, field_of_view, angle_min)
        corners = self.extractor.get_corners()

        self.extractor.plot_corners(path, ranges)
        #self.extractor.segment_detector.plot_segments_and_scan(path)

        # Publish the keypoints as markers
        keypoints_markers = self.generate_markers_from_keypoints(corners, [1.0, 0.0, 0.0], 0, scan_time)
        self.generate_poses_from_keypoint(corners, scan_time)
        self.keypoints_publisher.publish(keypoints_markers)
        rospy.loginfo("Extracted keypoints: %d", len(corners))
        #rospy.signal_shutdown("Shutting down gracefully.")
        

    def generate_poses_from_keypoint(self, keypoints, timestamp):
        """Generates poses from the keypoints given the color and the base index"""

        poses = PoseArray()
        poses.header.frame_id = "velodyne"
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
            poses.poses.append(pose)

        self.corner_poses_publisher.publish(poses)

    def generate_markers_from_keypoints(self, keypoints, color, base_idx, timestamp):
        """Generates markers from the keypoints given the color and the base index"""

        markers = MarkerArray()
        for idx, keypoint in enumerate(keypoints):
            marker = Marker()
            marker.header.frame_id = "velodyne"
            marker.header.stamp = timestamp
            marker.id = idx + base_idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(keypoint.x)
            marker.pose.position.y = float(keypoint.y)
            #rospy.loginfo("Corner Position:", keypoint.x, keypoint.y)
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
            marker.lifetime = rospy.Duration(1.0)

            markers.markers.append(marker)

        return markers

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
        """Converts polar coordinates to cartesian"""
        x = range * np.cos(angle)
        y = range * np.sin(angle)
        return np.array([x, y])

    def cartesian_to_polar(self, x, y):
        """Converts cartesian coordinates to polar"""
        range = np.sqrt(x**2 + y**2)
        angle = np.arctan2(y, x)
        return np.array([range, angle])

def main():
    slam_node = MyCornerExtractor()
    rospy.loginfo("SLAM node started")

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down SLAM node")

if __name__ == "__main__":
    main()