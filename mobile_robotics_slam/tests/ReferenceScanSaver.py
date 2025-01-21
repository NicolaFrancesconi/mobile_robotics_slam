import rclpy
import os
import sys
import signal
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry


# adding localization_lib to the system path
path = __file__
file_location_subfolders = 3 #Number of folder to go up to reach root of package
for _ in range(file_location_subfolders):
    path = os.path.dirname(path)
sys.path.insert(0, path)


class ReferenceScanGenerator(Node):
    def __init__(self):
        super().__init__("Scan_Saver", parameter_overrides=[])
        self.scan_subscriber = self.create_subscription(LaserScan, "/diff_drive/scan", self.scan_callback, 10)
        
    def scan_callback(self, msg: LaserScan):
        # Save the Scan Ranges into a txt file
        with open(os.path.join(path, "example_scans", "scan1.txt"), "w") as f:
            for i in range(len(msg.ranges)):
                string = str(msg.ranges[i]) + "\n" # Save only the ranges
                #string = str(msg.ranges[i]) + " " + str(msg.intensities[i]) + "\n" # Save the ranges and intensities
                f.write(string)
        print("Reference Scan Saved")
        self.destroy_node()
        rclpy.shutdown()
        exit()
    
    def signal_handler (self, sig, frame):
        self.destroy_node()
        rclpy.shutdown()
        
        

def main(args=None):
    rclpy.init(args=args)

    scan_generator = ReferenceScanGenerator()
    signal.signal(signal.SIGINT, scan_generator.signal_handler )

    try:
        rclpy.spin(scan_generator)
    except KeyboardInterrupt:
        pass
    finally:
        scan_generator.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
