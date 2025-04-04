import numpy as np
import os
import sys

path = __file__
file_location_subfolders = 3 #Number of folder to go up to reach root of package
for _ in range(file_location_subfolders):
    path = os.path.dirname(path)
sys.path.insert(0, path)

from mobile_robotics_slam.Extractors.Reflectors.ReflectorExtractor import ReflectorExtractor


#Test Using Data from intensity_scan.txt

# Load Data
data = np.loadtxt(os.path.join(path, "example_scans", "intensity_scan.txt"))
scan_ranges = data[:, 0]
scan_intensities = data[:, 1]
angle_min = -np.pi
field_of_view = 2*np.pi

# Extract Reflectors
reflector_extractor = ReflectorExtractor()
reflector_extractor.max_range_extraction = 30


robot_pose = np.zeros(3)
reflector_extractor.extract_reflectors(scan_ranges, scan_intensities, field_of_view, angle_min, robot_pose)
landmarks = reflector_extractor.get_reflectors()
positions = np.array([landmark.get_position() for landmark in landmarks])
print("Positions: ", positions)
reflector_extractor.plot_reflectors()

