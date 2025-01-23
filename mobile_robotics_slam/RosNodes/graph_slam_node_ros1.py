import rospy
import os
import sys
import signal
import matplotlib.pyplot as plt

import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import time
import message_filters

# Necessary to run the script from visual studio code
path = __file__
file_location_subfolders = 3 #Number of folder to go up to reach root of package
for _ in range(file_location_subfolders):
    path = os.path.dirname(path)
sys.path.insert(0, path)

from mobile_robotics_slam.Extractors.Reflectors.ReflectorExtractor import ReflectorExtractor
from mobile_robotics_slam.Extractors.Corners.CornerExtractor import CornerExtractor
from mobile_robotics_slam.GraphHandler.GTSAMGraphHandler import GraphHandler as GTSAMGraphHandler
from mobile_robotics_slam.MotionCompensation.MotionCompensation import LaserMotionCompensator as MotionCompensator
from mobile_robotics_slam.ICP.ICP_SVD import icp
from mobile_robotics_slam.MapGenerator.OnlineMap import DynamicMapUpdater


DISTANCE_THRESHOLD = 0.2
ROTATION_THRESHOLD = np.deg2rad(3)
EXTRACT_CORNER = False
EXTRACT_REFLECTORS = True

def remove_png_files(folder_path):
    try:
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".png"):
                file_path = os.path.join(folder_path, file_name)
                os.remove(file_path)
                print(f"Removed: {file_path}")
    except Exception as e:
        print(f"Error: {e}")


class GraphSlamNode:

    ##################################################################
    # Initialization
    ##################################################################
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node("graph_slam_node", anonymous=True)
        remove_png_files(path)

        # Declare variables
        self.OdomInitialPose = np.array([None, None, None])
        self.OdomLastNodePose = np.zeros(3)
        self.OptimizedLastNodePose = np.zeros(3)
        self.OptimizedLastNodeScan = None
        self.H_RL = self.pose_to_transform([-0.109, 0, 0]) # Set Laser frame position wrt Robot Frame (x,y, theta)
        self.first_pose_added = False
        self.new_pose_added = False
        self.add_last_pose = False
        self.reflector_extractor = ReflectorExtractor()
        self.corner_extractor = CornerExtractor()
        self.setup_extractor_parameters()

        self.graph_handler = GTSAMGraphHandler()
        #self.graph_handler = g2oGraphHandler()
    
        self.odom_trajectory = []

        self.odom_sub = message_filters.Subscriber("/odometry/filtered", Odometry)
        self.scan_sub = message_filters.Subscriber("/scan", LaserScan)

        self.dynamic_map = DynamicMapUpdater()
        self.dynamic_map.start()
        self.map_update_timer = rospy.Timer(rospy.Duration(3), self.map_timer_callback)


        # Approximate time synchronizer
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.odom_sub, self.scan_sub],
            queue_size=10,
            slop=0.02  # Max difference between timestamps
        )
        self.sync.registerCallback(self.synchronized_callback)

        


    def setup_extractor_parameters(self):
        # Set the parameters of the Corner Extractor
        min_corner_angle = 85
        max_corner_angle = 95
        max_intersecton_distance = 0.4
        self.corner_extractor.set_corner_params(max_intersecton_distance, min_corner_angle, max_corner_angle)

        # Set the parameters of the Adaptive Segment Detector
        sigma_ranges = 0.20
        lambda_angle = 10
        merge_distance = 0.15
        min_points_density = 5
        min_segment_length = 0.2
        self.corner_extractor.set_detector_params(sigma_ranges, lambda_angle, merge_distance, min_points_density, min_segment_length)

        # Set the parameters of the Segment Handler
        epsilon = 0.12
        min_density_after_segmentation = 5
        min_length_after_segmentation = 0.2
        self.corner_extractor.set_handler_params(epsilon, min_density_after_segmentation, min_length_after_segmentation)

    def map_timer_callback(self, event):
        if not self.new_pose_added:
            return
        poses, pointclouds, landmarks = self.graph_handler.get_optimized_poses_and_landmarks()
        cartesian_points = []
        robot_poses = []
        
        for pose, pointcloud in zip(poses, pointclouds):
            angle = np.linspace(-np.pi, np.pi, len(pointcloud))
            x = pose[0] + pointcloud * np.cos(angle + pose[2])
            y = pose[1] + pointcloud * np.sin(angle + pose[2])
            cartesian_points.extend(np.vstack((x, y)).T)
            # In poses there is laser pose, we need to convert it to robot pose
            robot_poses.append(self.transform_to_pose(self.pose_to_transform(pose)@np.linalg.inv(self.H_RL)))
        
        self.dynamic_map.add_data(robot_poses, landmarks, cartesian_points)
        self.new_pose_added = False

    def compute_homo_transform(self, pose1, pose2):
        T1 = self.pose_to_transform(pose1)
        T2 = self.pose_to_transform(pose2)
        H = np.linalg.inv(T1)@T2
        distance = np.linalg.norm(H[0:2, 2])
        rotation = np.abs(np.arctan2(H[1,0], H[0,0]))

        return H, distance, rotation

    def pose_to_transform(self, pose):
        """Given a pose [x,y,theta] it returns the Homogeneous
        transform T of the pose"""
        cos = np.cos(pose[2])
        sin = np.sin(pose[2])
        dx = pose[0]
        dy = pose[1]
        T = np.array([[cos, -sin, dx],
                      [sin, cos , dy],
                      [0  , 0   , 1 ]])
        return T
    
    def transform_to_pose(self, T):
        theta = np.arctan2(T[1,0], T[0,0])
        x = T[0, 2]
        y = T[1, 2]
        return np.array([x,y,theta])
    
    def add_first_pose(self, odom: Odometry, scan: LaserScan):
        """Add the first pose to the graph"""
        self.OdomReference = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, self.quaternion_to_euler(odom.pose.pose.orientation.x,odom.pose.pose.orientation.y,odom.pose.pose.orientation.z,odom.pose.pose.orientation.w)])
        self.OdomLastNodePose = np.zeros(3) # Initialize First Pose as origin (0,0,0)
        self.OptimizedLastNodePose = np.zeros(3) # Initialize First Pose as origin (0,0,0)
        self.OptimizedLastNodeScan = np.array(scan.ranges)
        laser_estimated_pose = self.transform_to_pose(self.H_RL) # Initialize Laser Pose as Laser Frame wrt Robot Frame
        self.odom_trajectory.append(np.copy(self.OdomLastNodePose))
        reflectors = []
        corners = []

        if EXTRACT_CORNER:
            corners = self.extract_corners(scan, laser_estimated_pose)
        if EXTRACT_REFLECTORS:
            reflectors = self.extract_reflectors(scan, laser_estimated_pose)
        
        landmarks = reflectors + corners
        laser_optimized_pose = self.graph_handler.add_to_graph(laser_estimated_pose, np.array(scan.ranges), landmarks)
        T_laser_optimized = self.pose_to_transform(laser_optimized_pose)
        self.OptimizedLastNodePose = self.transform_to_pose(T_laser_optimized@np.linalg.inv(self.H_RL))
        self.new_pose_added = True


    def synchronized_callback(self, odom: Odometry, scan: LaserScan):

        start_time = time.time()

        # Store ODOM and REAL pose for Visualization
        if not self.first_pose_added:
            self.add_first_pose(odom, scan)
            self.first_pose_added = True
            return
        
        odom_pose = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, self.quaternion_to_euler(odom.pose.pose.orientation.x,odom.pose.pose.orientation.y,odom.pose.pose.orientation.z,odom.pose.pose.orientation.w)])
         
        #Estimate Motion of Robot Using Odometry
        H_robot_odom, travel_distance, rotation = self.compute_homo_transform(self.OdomReference, odom_pose)
        
        #If Motion Higher than THRESHOLD correct it using ICP
        if travel_distance > DISTANCE_THRESHOLD or rotation > ROTATION_THRESHOLD or self.add_last_pose:
            
            #ICP is WRT Laser Frame so converti H_robot into H_laser
            H_laser_odom = np.dot(H_robot_odom,self.H_RL)  # Homogeneous Transform of LASER due to Odometry estimate)
            
            #Prepare previous and current scan for ICP
            angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
            cos = np.cos(angles)
            sin = np.sin(angles)
            previous_scan = np.copy(self.OptimizedLastNodeScan)
            current_scan = np.array(scan.ranges)
            current_points = np.vstack((current_scan*cos, current_scan*sin)).T
            previous_points = np.vstack((previous_scan*cos, previous_scan*sin)).T

            #Perform ICP to estimate a better H_laser and thus H_robot
            H_laser_icp = icp(current_points, previous_points, init_transform=H_laser_odom, downsample=4, max_iterations=30, max_range=15)
            H_robot_icp = H_laser_icp@(np.linalg.inv(self.H_RL))

            #Update the estimated pose of the laser given the ICP result
            Tr = self.pose_to_transform(self.OptimizedLastNodePose)
            laser_estimated_pose = self.transform_to_pose((Tr@H_robot_icp)@self.H_RL)
                
            #Extract Landmarks from the scan wrt Laser Frame
            reflectors = []
            corners = []
            if EXTRACT_CORNER:
                corners = self.extract_corners(scan, laser_estimated_pose)
            if EXTRACT_REFLECTORS:
                reflectors = self.extract_reflectors(scan, laser_estimated_pose)
            landmarks = reflectors + corners
            
            #Add the estimated of laser pose to the graph and get the optimized pose of laser
            laser_optimized_pose = self.graph_handler.add_to_graph(laser_estimated_pose, np.array(scan.ranges), landmarks)
            T_laser_optimized = self.pose_to_transform(laser_optimized_pose)

            #Update the optimized pose of the robot given the optimized pose of the laser
            self.OptimizedLastNodePose = self.transform_to_pose(T_laser_optimized@np.linalg.inv(self.H_RL))
            
            #Store Data about the Node for the next iteration
            Tr_odom = self.pose_to_transform(self.OdomLastNodePose)
            self.OdomLastNodePose = self.transform_to_pose(Tr_odom@H_robot_odom)
            self.OdomReference = np.copy(odom_pose)
            self.OptimizedLastNodeScan = np.copy(scan.ranges)

            #Store Data for Visualization
            self.odom_trajectory.append(np.copy(self.OdomLastNodePose))

            self.new_pose_added = True
            

            print("\n\nTime For Processing: ", time.time() - start_time)
            print(f"Odom Estimate: {self.OdomLastNodePose}")      
            print(f"Estimated Pose: {self.OptimizedLastNodePose}")

            if self.add_last_pose:
                print("Last Pose Added")
                self.add_last_pose = False
                self.save_data()

    def save_data(self):
        poses, _, landmarks = self.graph_handler.get_optimized_poses_and_landmarks()
        robot_trajectory = []
        for pose in poses:
            robot_trajectory.append(self.transform_to_pose(self.pose_to_transform(pose)@np.linalg.inv(self.H_RL)))
        robot_trajectory = np.array(robot_trajectory)
        
        # If directory does not exist, create it
        save_path = os.path.join(path, "trajectory_data")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Save the data
        np.savetxt(os.path.join(save_path, "robot_optimized.txt"), robot_trajectory)
        np.savetxt(os.path.join(save_path,  "odom_trajectory.txt"), np.array(self.odom_trajectory))
        np.savetxt(os.path.join(save_path,  "landmarks.txt"), np.array(landmarks))

        print(f"Saved Data in Folder: {save_path} ")

    def extract_reflectors(self, scan: LaserScan, laser_estimated_pose):
        pointcloud = np.array(scan.ranges)
        intensities = np.array(scan.intensities)
        field_of_view = scan.angle_max - scan.angle_min
        angle_min = scan.angle_min
        self.reflector_extractor.extract_reflectors(pointcloud, intensities, field_of_view, angle_min, laser_estimated_pose)
        keypoints = self.reflector_extractor.get_reflectors()
        return keypoints

    def extract_corners(self, scan: LaserScan, laser_estimated_pose):
        pointcloud = np.array(scan.ranges)
        field_of_view = scan.angle_max - scan.angle_min
        angle_min = scan.angle_min
        self.corner_extractor.extract_corners(pointcloud, field_of_view, angle_min, laser_estimated_pose)
        keypoints = self.corner_extractor.get_corners()
        return keypoints
     

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
    

    def signal_handler(self,sig, frame):
        self.add_last_pose = True

        
def main():
    print("SLAM node started")
    slam_node = GraphSlamNode()
    signal.signal(signal.SIGINT, slam_node.signal_handler)

    try:
        rospy.spin()  # Keep the node running and process callbacks
    except Exception as e:
        print(f"An exception occurred: {e}")
    finally:
        slam_node.destroy_node()
        rospy.loginfo("Node shut down complete")

if __name__ == '__main__':
    main()
