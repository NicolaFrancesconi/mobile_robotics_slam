import rclpy
import os
import sys
import signal
from rclpy.node import Node
import matplotlib.pyplot as plt

import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import time
import message_filters

path = __file__
file_location_subfolders = 3 #Number of folder to go up to reach root of package
for _ in range(file_location_subfolders):
    path = os.path.dirname(path)
sys.path.insert(0, path)



from mobile_robotics_slam.Extractors.Reflectors.ReflectorExtractor import ReflectorExtractor
from mobile_robotics_slam.Extractors.Corners.CornerExtractor import CornerExtractor
from mobile_robotics_slam.GraphHandler.g2oGraphHandler import GraphHandler as g2oGraphHandler
from mobile_robotics_slam.GraphHandler.GTSAMGraphHandler import GraphHandler as GTSAMGraphHandler
from mobile_robotics_slam.ICP.ICP_SVD import icp


DISTANCE_THRESHOLD = 0.4
ROTATION_THRESHOLD = np.deg2rad(5)



class GraphSLamNode(Node):

##################################################################
# Initialization
##################################################################
    def __init__(self):
        super().__init__("graph_slam_node", parameter_overrides=[]) 

        # Declare variables
        self.OdomLastNodePose = np.array([None, None, None])
        self.OptimizedLastNodePose = np.array([None, None, None])
        self.OptimizedLastNodeScan = None
        self.H_RL = self.pose_to_transform([-0.109, 0, 0]) # Set Laser frame position wrt Robot Frame (x,y, theta)
        self.first_pose_added = False
        self.reflector_extractor = ReflectorExtractor()
        self.corner_extractor = CornerExtractor()
        self.setup_extractor_parameters()

        self.graph_handler = GTSAMGraphHandler()
        #self.graph_handler = g2oGraphHandler()

        self.real_trajectory = []
        self.odom_trajectory = []

        self.unoptimized_graph = UnoptimizedGraph()

        self.odom_sub = message_filters.Subscriber(self, Odometry, "/dingo/odometry")
        self.scan_sub = message_filters.Subscriber(self, LaserScan, "/diff_drive/scan")
        self.real_pose_sub = message_filters.Subscriber(self, Odometry, "/diff_drive/real_pose")

        # Approximate time synchronizer
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.odom_sub, self.scan_sub, self.real_pose_sub], 
            queue_size=30, 
            slop=0.0001 #Max difference between timestamps
        )
        self.sync.registerCallback(self.synchronized_callback)


    def setup_extractor_parameters(self):
        # Set the parameters of the Corner Extractor
        min_corner_angle = 85
        max_corner_angle = 95
        max_intersecton_distance = 0.5
        self.corner_extractor.set_corner_params(max_intersecton_distance, min_corner_angle, max_corner_angle)

        # Set the parameters of the Adaptive Segment Detector
        sigma_ranges = 0.15
        lambda_angle = 10
        merge_distance = 0.07
        min_points_density = 10
        min_segment_length = 0.5
        self.corner_extractor.set_detector_params(sigma_ranges, lambda_angle, merge_distance, min_points_density, min_segment_length)

        # Set the parameters of the Segment Handler
        epsilon = 0.1
        min_density_after_segmentation = 12
        min_length_after_segmentation = 0.3
        self.corner_extractor.set_handler_params(epsilon, min_density_after_segmentation, min_length_after_segmentation)

    def compute_homo_transform(self, pose1, pose2):
        """
            Given two poses (x,y,theta) return the homogeneous transformation between them
            Input:  pose1: (x,y,theta) of previous pose
                    pose2: (x,y,theta) of new pose
            
            Output: H: Homogeneous transformation between the two poses (3x3 matrix)
                    distance: linear distance between the two poses 
                    rotation: angular rotation between two poses
        """
        T1 = self.pose_to_transform(pose1)
        T2 = self.pose_to_transform(pose2)
        H = np.linalg.inv(T1)@T2
        distance = np.linalg.norm(H[0:2, 2])
        rotation = np.abs(np.arctan2(H[1,0], H[0,0]))

        return H, distance, rotation

    def pose_to_transform(self, pose):
        """Given a pose [x,y,theta] it returns the Homogeneous with respect to origin (0,0,0)
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
    
    def add_first_pose(self, odom: Odometry, scan: LaserScan, real: Odometry):
        self.OdomLastNodePose = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, self.quaternion_to_euler(odom.pose.pose.orientation.x,odom.pose.pose.orientation.y,odom.pose.pose.orientation.z,odom.pose.pose.orientation.w)])
        self.OptimizedLastNodePose = np.zeros(3) # Initialize First Pose as origin (0,0,0)
        self.OptimizedLastNodeScan = np.array(scan.ranges)
        robot_estimated_pose = np.zeros(3)
        laser_estimated_pose = self.transform_to_pose(self.H_RL) # Initialize Laser Pose as Laser Frame wrt Robot Frame
        
        reflectors = []
        corners = []
        reflectors = self.extract_reflectors(scan, laser_estimated_pose)
        #corners = self.extract_corners(scan, laser_estimated_pose)
        landmarks = reflectors + corners
        laser_optimized_pose = self.graph_handler.add_to_graph(laser_estimated_pose, np.array(scan.ranges), landmarks)
        T_laser_optimized = self.pose_to_transform(laser_optimized_pose)
        self.OptimizedLastNodePose = self.transform_to_pose(T_laser_optimized@np.linalg.inv(self.H_RL))


    def synchronized_callback(self, odom: Odometry, scan: LaserScan, real: Odometry):
        start_time = time.time()

        # Store ODOM and REAL pose for Visualization
        real_pose = np.array([real.pose.pose.position.x, real.pose.pose.position.y, self.quaternion_to_euler(real.pose.pose.orientation.x,real.pose.pose.orientation.y,real.pose.pose.orientation.z,real.pose.pose.orientation.w)])
        odom_pose = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, self.quaternion_to_euler(odom.pose.pose.orientation.x,odom.pose.pose.orientation.y,odom.pose.pose.orientation.z,odom.pose.pose.orientation.w)])
        self.odom_trajectory.append(np.copy(odom_pose))
        self.real_trajectory.append(np.copy(real_pose))

        if not self.first_pose_added:
            self.add_first_pose(odom, scan, real)
            self.first_pose_added = True
            return
        
        #Estimate Motion of Robot Using Odometry
        H_robot_odom, travel_distance, rotation = self.compute_homo_transform(self.OdomLastNodePose, odom_pose)
        
        #If Motion Higher than THRESHOLD correct it using ICP
        if travel_distance > DISTANCE_THRESHOLD or rotation > ROTATION_THRESHOLD:
            
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

            #Update the estimated pose of the robot and laser
            Tr = self.pose_to_transform(self.OptimizedLastNodePose)
            Tr = Tr@H_robot_icp
            robot_estimated_pose = self.transform_to_pose(Tr)
            laser_estimated_pose = self.transform_to_pose(Tr@self.H_RL)
                
            #Extract Landmarks from the scan wrt Laser Frame
            reflectors = []
            corners = []
            reflectors = self.extract_reflectors(scan, laser_estimated_pose)
            #corners = self.extract_corners(scan, laser_estimated_pose)
            landmarks = reflectors + corners
            
            #Add the estimated of laser pose to the graph and get the optimized pose of laser
            laser_optimized_pose = self.graph_handler.add_to_graph(laser_estimated_pose, np.array(scan.ranges), landmarks)
            T_laser_optimized = self.pose_to_transform(laser_optimized_pose)

            #Update the optimized pose of the robot given the optimized pose of the laser
            self.OptimizedLastNodePose = self.transform_to_pose(T_laser_optimized@np.linalg.inv(self.H_RL))
            
            #Store Data about the Node for the next iteration
            self.OdomLastNodePose = np.copy(odom_pose)
            self.OptimizedLastNodeScan = np.copy(scan.ranges)

            #Store Data for Visualization
            self.odom_trajectory.append(np.copy(odom_pose))
            self.real_trajectory.append(np.copy(real_pose))
            self.unoptimized_graph.add_pose(odom_pose, np.array(scan.ranges), landmarks)

            
            print("\n\nTime For Processing: ", time.time() - start_time)
            print(f"Odom Estimate: {self.OdomLastNodePose}")      
            print(f"Estimated Pose: {self.OptimizedLastNodePose}")
            print(f"Real Pose: {real_pose}")                           




    def extract_reflectors(self, scan: LaserScan, scan_frame_pose):
        """Extracts the reflectors from the scan and returns them as keypoints
            Input:  scan: LaserScan message
                    scan_frame_pose: Pose of the laser frame in global coordinates
            Output: keypoints: list of reflectors keypoint as Object with global position
        """
        pointcloud = np.array(scan.ranges)
        intensities = np.array(scan.intensities)
        field_of_view = scan.angle_max - scan.angle_min
        angle_min = scan.angle_min
        self.reflector_extractor.extract_reflectors(pointcloud, intensities, field_of_view, angle_min, scan_frame_pose)
        keypoints = self.reflector_extractor.get_reflectors()
        return keypoints


    def extract_corners(self, scan: LaserScan, scan_frame_pose):
        """Extracts the corners from the scan and returns them as keypoints
            Input:  scan: LaserScan message
                    scan_frame_pose: Pose of the laser frame in global coordinates
            Output: keypoints: list of corners keypoint as Object with global position
        """
        pointcloud = np.array(scan.ranges)
        field_of_view = scan.angle_max - scan.angle_min
        angle_min = scan.angle_min
        self.corner_extractor.extract_corners(pointcloud, field_of_view, angle_min, scan_frame_pose)
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

def save_data(node : GraphSLamNode):
    # Save the optimized poses
    pass #TODO: Implement this function to store data
    
        
def main(args=None):
    rclpy.init(args=args)

    print("SLAM node started")

    slam_node = GraphSLamNode()

    try:
        rclpy.spin(slam_node)
    except KeyboardInterrupt:
        pass  # Graceful handling of Ctrl+C
    finally:
        save_data(slam_node)
        slam_node.destroy_node()
        rclpy.try_shutdown()


class map_pose:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.point_cloud = []
        self.landmarks = []

class UnoptimizedGraph:
    def __init__(self):
        self.poses = []

    def add_pose(self, robot_pose, pointcloud, landmarks):
        pose = map_pose()
        pose.x = robot_pose[0]
        pose.y = robot_pose[1]
        pose.theta = robot_pose[2]
        pose.point_cloud = pointcloud
        pose.landmarks = [landmark.get_position() for landmark in landmarks]
        self.poses.append(pose)

    def generate_map(self):
        map = []
        poses = []
        landmarks = []
        for pose in self.poses:
            ranges = pose.point_cloud
            angles = np.linspace(-np.pi, np.pi, len(ranges))
            x = pose.x + ranges * np.cos(angles + pose.theta)
            y = pose.y + ranges * np.sin(angles + pose.theta)
            map.extend(np.vstack((x, y)).T)
            poses.append([pose.x, pose.y, pose.theta])
            landmarks.extend(pose.landmarks)



        map = np.array(map)
        poses = np.array(poses)
        landmarks = np.array(landmarks)

        plt.figure()
        plt.title("Unoptimized Map")
        plt.scatter(map[:, 0], map[:, 1], c='g', s=1)
        plt.plot(poses[:, 0], poses[:, 1], "b")
        if len(landmarks) > 0:
            plt.scatter(landmarks[:, 0], landmarks[:, 1], c="r")
        
        plt.axis('equal')
        plt.legend(['Pointcloud', 'Pose', '   Landmark'])
        plt.show()

if __name__ == "__main__":
    main()

