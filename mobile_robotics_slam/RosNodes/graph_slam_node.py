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
from mobile_robotics_slam.GraphHandler.GraphHandler import GraphHandler
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
        self.robot_pose_x = None
        self.robot_pose_y = None
        self.robot_pose_phi = None
        self.displacement = np.array([0.0, 0.0, 0.0])
        self.first_pose_added = False
        self.reflector_extractor = ReflectorExtractor()
        self.corner_extractor = CornerExtractor()
        self.setup_extractor_parameters()

        self.graph_handler = GraphHandler()

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


    def synchronized_callback(self, odom: Odometry, scan: LaserScan, real: Odometry):
        if (self.OptimizedLastNodePose[0] is None) or (self.OdomLastNodePose[0] is None):
            self.OdomLastNodePose = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, self.quaternion_to_euler(odom.pose.pose.orientation.x,odom.pose.pose.orientation.y,odom.pose.pose.orientation.z,odom.pose.pose.orientation.w)])
            #self.OdomLastNodePose = np.array([real.pose.pose.position.x, real.pose.pose.position.y, self.quaternion_to_euler(real.pose.pose.orientation.x,real.pose.pose.orientation.y,real.pose.pose.orientation.z,real.pose.pose.orientation.w)])
            
            self.OptimizedLastNodePose = np.zeros(3)
            robot_estimated_pose = np.zeros(3)
        

        # Store ODOM and REAL pose for Visualization
        real_pose = np.array([real.pose.pose.position.x, real.pose.pose.position.y, self.quaternion_to_euler(real.pose.pose.orientation.x,real.pose.pose.orientation.y,real.pose.pose.orientation.z,real.pose.pose.orientation.w)])
        odom_pose = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, self.quaternion_to_euler(odom.pose.pose.orientation.x,odom.pose.pose.orientation.y,odom.pose.pose.orientation.z,odom.pose.pose.orientation.w)])
        #odom_pose = np.copy(real_pose)

        Ho, travel_distance, rotation = self.compute_homo_transform(self.OdomLastNodePose, odom_pose)
        
        if travel_distance > DISTANCE_THRESHOLD or rotation > ROTATION_THRESHOLD or not self.first_pose_added:
            start_time = time.time()
            if  self.first_pose_added:
                angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
                cos = np.cos(angles)
                sin = np.sin(angles)
                previous_scan = self.unoptimized_graph.poses[-1].point_cloud
                current_scan = scan.ranges
                Tr = self.pose_to_transform(self.OptimizedLastNodePose)
                #robot_estimated_pose=self.transform_to_pose(Tr@Ho)
                current_points = np.vstack((current_scan*cos, current_scan*sin)).T
                previous_points = np.vstack((previous_scan*cos, previous_scan*sin)).T
                times = time.time()
                H_icp = icp(current_points, previous_points, init_transform=Ho, downsample=8, max_iterations=30, max_range=15)
                print("Time For ICP", time.time() - times)
                #print("Inverse\n", np.linalg.inv(H_icp))
                
                robot_estimated_pose = self.transform_to_pose(Tr@H_icp)
                #robot_estimated_pose = self.transform_to_pose(Tr@Ho)
                
            
            self.first_pose_added = True

            reflectors = []
            corners = []

            reflectors = self.extract_reflectors(scan, robot_estimated_pose)
            #corners = self.extract_corners(scan, robot_estimated_pose)

            landmarks = reflectors + corners
            

            self.OdomLastNodePose = np.copy(odom_pose)
            self.OptimizedLastNodePose = np.copy(robot_estimated_pose)
            self.OptimizedLastNodePose = self.graph_handler.add_to_graph(robot_estimated_pose, np.array(scan.ranges), landmarks)
            

            self.odom_trajectory.append(np.copy(odom_pose))
            self.real_trajectory.append(np.copy(real_pose))
            self.unoptimized_graph.add_pose(odom_pose, np.array(scan.ranges), landmarks)

            print("\n\nTime For Processing: ", time.time() - start_time)
            print(f"Odom Estimate: {self.OdomLastNodePose}")      
            print(f"Estimated Pose: {robot_estimated_pose}")
            print(f"Real Pose: {real_pose}")                           


    def points_3d_from_scan_and_pose(self, scan, robot_estimated_pose=np.zeros(3), max_range=np.inf, downsample=1):
        angles = np.linspace(-np.pi, np.pi, len(scan))
        scan = np.array(scan)
        ranges = scan[scan <= max_range][::downsample]
        angles = angles[scan <= max_range][::downsample]
        x = ranges * np.cos(angles + robot_estimated_pose[2]) + robot_estimated_pose[0]
        y = ranges * np.sin(angles + robot_estimated_pose[2]) + robot_estimated_pose[1]
        return np.vstack((x, y)).T


    def extract_reflectors(self, scan: LaserScan, robot_estimated_pose):
        pointcloud = np.array(scan.ranges)
        intensities = np.array(scan.intensities)
        field_of_view = scan.angle_max - scan.angle_min
        angle_min = scan.angle_min
        self.reflector_extractor.extract_reflectors(pointcloud, intensities, field_of_view, angle_min, robot_estimated_pose)
        keypoints = self.reflector_extractor.get_reflectors()
        return keypoints


    def extract_corners(self, scan: LaserScan, robot_estimated_pose):
        pointcloud = np.array(scan.ranges)
        field_of_view = scan.angle_max - scan.angle_min
        angle_min = scan.angle_min
        self.corner_extractor.extract_corners(pointcloud, field_of_view, angle_min, robot_estimated_pose)
        keypoints = self.corner_extractor.get_corners()
        return keypoints

    def compute_travel_distance_and_rotation(self, displacement):
        dx, dy, dphi = displacement
        travel_distance = np.linalg.norm([dx, dy])
        rotation = np.abs(dphi)
        return travel_distance, rotation

    def compute_robot_estimate(self, displacement, previous_pose):
        prev_x, prev_y, prev_phi = previous_pose
        dx, dy, dphi = displacement
        robot_pose_x = prev_x + (dx * np.cos(prev_phi) - dy * np.sin(prev_phi))
        robot_pose_y = prev_y + (dy * np.cos(prev_phi) + dx * np.sin(prev_phi))
        robot_pose_phi = prev_phi + dphi
        return np.array([robot_pose_x, robot_pose_y, robot_pose_phi])
    
    def compute_odometry_displacement(self, x, y, phi):
        dx = x - self.OdomLastNodePose[0]
        dy = y - self.OdomLastNodePose[1]
        dphi = (phi - self.OdomLastNodePose[2] + np.pi) % (2 * np.pi) - np.pi
        cos = np.cos(self.OdomLastNodePose[2])
        sin = np.sin(self.OdomLastNodePose[2])
        dx_local = cos * dx + sin * dy
        dy_local = -sin * dx + cos * dy
        return np.array([dx_local, dy_local, dphi])
        
        
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
        x_lidar = (x - self.robot_pose_x) * np.cos(self.robot_pose_phi) + (y - self.robot_pose_y) * np.sin(self.robot_pose_phi) # type: ignore
        y_lidar = -(x - self.robot_pose_x) * np.sin(self.robot_pose_phi) + (y - self.robot_pose_y) * np.cos(self.robot_pose_phi) # type: ignore
        return [x_lidar, y_lidar]

    def set_robot_real_pose(self, pose_x, pose_y, phi):
        """Set the real robot pose"""
        self.robot_pose_x = pose_x
        self.robot_pose_y = pose_y
        self.robot_pose_phi = phi

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
    

    def signal_handler(self, sig, frame):
        # Generate the map of the environment given the optimized graph
        self.unoptimized_graph.generate_map()
        self.graph_handler.generate_map(real_trajectory=self.real_trajectory, odom_trajectory=self.odom_trajectory)
        self.graph_handler.generate_dynamic_map()
        self.graph_handler.dynamic_map.stop()
    
        self.destroy_node()
        rclpy.try_shutdown()

        
def main(args=None):
    rclpy.init(args=args)

    print("SLAM node started")

    slam_node = GraphSLamNode()
    signal.signal(signal.SIGINT, slam_node.signal_handler)

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

