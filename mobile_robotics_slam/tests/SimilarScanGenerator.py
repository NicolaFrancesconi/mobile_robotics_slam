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

from mobile_robotics_slam.Keypoint import Keypoint, KeypointList

SCAN_TO_GENERATE = 15

NOISE_TEST = False
NOISE_STANDARD_DEVIATION = 0.02

OVERSAMPLING_TEST = False
ADDING_ELEMENTS_BETWEEN_POINTS = 1

DOWNSAMPLING_TEST = True
REMOVING_ELEMENTS_BETWEEN_POINTS = 1

NEAREST_NEIGHBOR_MATCHING_DISTANCE = 0.05
NEAREST_NEIGHBOR_NEW_KEYPOINT_DISTANCE = 0.2



class ReferenceScanGenerator(Node):
    def __init__(self):
        super().__init__("Scan_Generator", parameter_overrides=[])
        # Read the reference scan from a file
        with open(os.path.join(path, "example_scans", "reference_scan.txt"), "r") as f:
            self.reference_scan = [float(line.strip()) for line in f]

        # Remove Inf values from the reference scan
        self.reference_scan = [scan if scan != float("inf") else 60.0 for scan in self.reference_scan]

        self.robot_pose = [0.0,0.0,0.0]
        self.number_of_scan_to_generate = SCAN_TO_GENERATE
        self.number_of_scan_generated = 0
        self.scan_publisher = self.create_publisher(LaserScan, "/diff_drive/scan", 10)
        self.pose_publisher = self.create_publisher(Odometry, "/diff_drive/real_pose", 10)

        self.received_keypoints_falko = [False for i in range(self.number_of_scan_to_generate+1)]
        self.received_keypoints_oc = [False for i in range(self.number_of_scan_to_generate+1)]
        self.received_keypoints_my = [False for i in range(self.number_of_scan_to_generate+1)]

        self.falko_repeatibility = []
        self.OC_repeatibility = []
        self.MY_repeatibility = []

        self.falko_reference_keypoints = KeypointList()
        self.OC_reference_keypoints = KeypointList()

        self.MY_KeyPoints_subscriber = self.create_subscription(MarkerArray, "/my_keypoints", lambda msg: self.keypoints_callback(msg, self.MY_repeatibility, self.received_keypoints_my), 10)
        self.FALKO_Keypoints_subscriber = self.create_subscription(MarkerArray, "/falko_keypoints", self.keypoints_callback, 10)
        self.OC_KeyPoints_subscriber = self.create_subscription(MarkerArray, "/oc_keypoints", self.keypoints_callback, 10)
        self.FALKO_Keypoints_subscriber = self.create_subscription(MarkerArray, "/falko_keypoints", lambda msg :self.keypoints_callback(msg, self.falko_keypoints, "FALKO"), 10)

        self.REFERENCE_PASSED = False
        

        
        
        

        self.generated_scan = [False for i in range(self.number_of_scan_to_generate)]
        self.publish_reference_scan()
        self.create_timer(0.1, self.timer_callback)

    
    def signal_handler (self, sig, frame):
        print("Repeatibility Test Finished")
        print("Repeatibility FALKO: ", sum(self.falko_repeatibility)/len(self.falko_repeatibility))
        print("Repeatibility OC: ", sum(self.OC_repeatibility)/len(self.OC_repeatibility))

        self.destroy_node()
        rclpy.shutdown()


    def keypoints_callback(self, msg: MarkerArray, name):
        pass


        
        

    def publish_reference_scan(self):
        ranges = self.reference_scan.copy()
        scan = LaserScan()
        scan.header.frame_id = "lidar_link"
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.angle_min = 0.0
        scan.angle_max = 2*np.pi
        scan.angle_increment = 2*np.pi/len(ranges) #type: ignore
        scan.range_min = 0.0
        scan.range_max = 60.0
        scan.ranges = ranges
        scan.intensities = [0.0 for i in range(len(ranges))] #type: ignore
        self.scan_publisher.publish(scan)
        pose = Odometry()
        pose.header.frame_id = "odom"
        self.pose_publisher.publish(pose)
        self.REFERENCE_PASSED = True



    def FALKO_Keypoints_callback(self, msg: MarkerArray):
        if self.falko_reference_keypoints.keypoints == []:
            for marker in msg.markers:
                self.falko_reference_keypoints.add_keypoint(marker.pose.position.x, marker.pose.position.y)
        else:
            keypoints = []
            for marker in msg.markers:
                keypoints.append([marker.pose.position.x, marker.pose.position.y])
            match = self.falko_reference_keypoints.match_keypoints_NN(keypoints, NEAREST_NEIGHBOR_MATCHING_DISTANCE, NEAREST_NEIGHBOR_NEW_KEYPOINT_DISTANCE)[0]
            if len(match) > 0:
                self.falko_repeatibility.append(self.evaluate_repeatability(len(match), len(self.falko_reference_keypoints.keypoints), len(keypoints)))
            else:
                self.falko_repeatibility.append(0.0)
        self.received_keypoints_falko[self.number_of_scan_generated] = True

            
            


    def OC_KeyPoints_callback(self, msg: MarkerArray):
        if self.OC_reference_keypoints.keypoints == []:
            for marker in msg.markers:
                self.OC_reference_keypoints.add_keypoint(marker.pose.position.x, marker.pose.position.y)
            self.received_keypoints_oc[self.number_of_scan_generated] = True
        else:
            keypoints = []
            for marker in msg.markers:
                keypoints.append([marker.pose.position.x, marker.pose.position.y])
            match = self.OC_reference_keypoints.match_keypoints_NN(keypoints, NEAREST_NEIGHBOR_MATCHING_DISTANCE, NEAREST_NEIGHBOR_NEW_KEYPOINT_DISTANCE)[0]
            if len(match) > 0:
                self.OC_repeatibility.append(self.evaluate_repeatability(len(match), len(self.OC_reference_keypoints.keypoints), len(keypoints)))
            else:
                self.OC_repeatibility.append(0.0)
            self.received_keypoints_oc[self.number_of_scan_generated] = True

            


    def generate_noisy_scan(self, standard_deviation):
        # add noise to the reference scan
        noisy_scan = self.reference_scan.copy()
        for i in range(len(noisy_scan)):
            noisy_scan[i] += np.random.normal(0, standard_deviation)
        return noisy_scan


    def evaluate_interpolation_point(self, point1, point2, thetha):
        delta_x = point2[0] - point1[0]
        delta_y = point2[1] - point1[1]
        if np.abs(point1[0] - point2[0]) < 1e-6:
            delta_x = np.sign(delta_x) * 1e-6

        
        m_points = delta_y / delta_x
        
        q_points = point1[1] - m_points * point1[0]

        m_angle = np.tan(thetha)
        q_angle = 0 #through the origin

        x_intersec = (q_angle - q_points) / (m_points - m_angle)
        y_intersec = m_points * x_intersec + q_points

        return np.linalg.norm([x_intersec, y_intersec])

    
    def generate_over_sampled_scan(self, adding_elemets_between_points): 
        scan = self.reference_scan.copy()
        over_sampled_scan = []
        angle_increment = 2*np.pi/len(scan)
        for i in range(len(scan)):
            j = 0
            point1 = np.array([scan[i] * np.cos(i*angle_increment), scan[i] * np.sin(i*angle_increment)])
            point2 = np.array([scan[(i+1)%len(scan)] * np.cos((i+1)*angle_increment), scan[(i+1)%len(scan)] * np.sin((i+1)*angle_increment)])
            # Interpolate (multipling_factor) points between point1 and point2
            over_sampled_scan.append(scan[i])
            theta = i*angle_increment
            while j < adding_elemets_between_points:
                theta_interp = theta + (j+1) * angle_increment/(adding_elemets_between_points+1)
                over_sampled_scan.append(self.evaluate_interpolation_point(point1, point2, theta_interp))
                j += 1
        return over_sampled_scan
             

    def generate_down_sampled_scan(self, removing_elements_between_points):
        scan = self.reference_scan.copy()
        down_sampled_scan = []
        for i in range(len(scan)):
            if i % (removing_elements_between_points+1) == 0:
                down_sampled_scan.append(scan[i])
        return down_sampled_scan


    def evaluate_repeatability(self, cardinality_match, cardinality_reference, cardinality_similar):
        """
        Repeatibility measure is given by the cardinality of the intersection of the two sets of keypoints
        divided by the minimum cardinality of the two sets of keypoints
        """
        return cardinality_match / min(cardinality_reference, cardinality_similar)


    def euler_to_quaternion(self, phi):
        """Converts euler angles to quaternion"""
        x = 0.0
        y = 0.0
        z = np.sin(phi / 2)
        w = np.cos(phi / 2)

        return x, y, z, w


    def timer_callback(self):
        ranges = self.reference_scan.copy()

        if NOISE_TEST:
            ranges = self.generate_noisy_scan(NOISE_STANDARD_DEVIATION)
        elif OVERSAMPLING_TEST:
            ranges = self.generate_over_sampled_scan(ADDING_ELEMENTS_BETWEEN_POINTS)
        elif DOWNSAMPLING_TEST:
            ranges = self.generate_down_sampled_scan(REMOVING_ELEMENTS_BETWEEN_POINTS)

        if self.number_of_scan_generated < self.number_of_scan_to_generate and self.REFERENCE_PASSED and self.received_keypoints_falko[self.number_of_scan_generated] and self.received_keypoints_oc[self.number_of_scan_generated]:
            scan = LaserScan()
            scan.header.frame_id = "lidar_link"
            scan.header.stamp = self.get_clock().now().to_msg()
            scan.angle_min = 0.0
            scan.angle_max = 2*np.pi
            scan.angle_increment = 2*np.pi/len(ranges) #type: ignore
            scan.range_min = 0.0
            scan.range_max = 60.0
            scan.ranges = ranges
            scan.intensities = [0.0 for i in range(len(ranges))] #type: ignore
        
            pose = Odometry()
            pose.header.frame_id = "odom"
            pose.pose.pose.position.x = self.robot_pose[0]
            pose.pose.pose.position.y = self.robot_pose[1]
            pose.pose.pose.position.z = 0.0
            pose.pose.pose.orientation.x, pose.pose.pose.orientation.y, pose.pose.pose.orientation.z, pose.pose.pose.orientation.w = self.euler_to_quaternion(self.robot_pose[2])
            self.number_of_scan_generated += 1
            self.pose_publisher.publish(pose)
            self.scan_publisher.publish(scan)
            
        # if all the elements in the received_keypoints_falko and received_keypoints_oc are True then the test is finished
        if all(self.received_keypoints_falko) and all(self.received_keypoints_oc):
            print("Repeatibility Test Finished")
            print("Repeatibility FALKO: ", sum(self.falko_repeatibility)/len(self.falko_repeatibility))
            print("Repeatibility OC: ", sum(self.OC_repeatibility)/len(self.OC_repeatibility))
            self.destroy_node()
            rclpy.shutdown()
            exit(0)




def main(args=None):
    rclpy.init(args=args)

    print("SCAN GENERATOR NODE")

    scan_generator = ReferenceScanGenerator()
    signal.signal(signal.SIGINT, scan_generator.signal_handler )
    

    try:
        rclpy.spin(scan_generator)
    except KeyboardInterrupt:
        pass
    finally:
        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        scan_generator.destroy_node()
        rclpy.try_shutdown()
        


if __name__ == "__main__":
    main()
