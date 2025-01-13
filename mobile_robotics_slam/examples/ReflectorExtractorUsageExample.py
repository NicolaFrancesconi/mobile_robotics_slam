import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "src", "application", "mobile_robotics_slam"))

from mobile_robotics_slam.Extractors.Reflectors.ReflectorExtractor import ReflectorExtractor


#Test Using Data from intensity_scan.txt

# Load Data
data = np.loadtxt('intensity_scan.txt')
scan_ranges = data[:, 0]
scan_intensities = data[:, 1]
angle_min = -np.pi/2
field_of_view = 2*np.pi

# Extract Reflectors
reflector_extractor = ReflectorExtractor()

robot_pose = np.array([1, 0, np.pi/3])
reflector_extractor.extract_reflectors(scan_ranges, scan_intensities, field_of_view, angle_min, robot_pose)
landmarks = reflector_extractor.get_reflectors()
positions = np.array([landmark.get_position() for landmark in landmarks])
print("Positions: ", positions)
reflector_extractor.plot_reflectors()

