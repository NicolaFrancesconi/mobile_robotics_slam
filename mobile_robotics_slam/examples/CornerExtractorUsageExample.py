import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
import open3d as o3d
from sklearn.metrics import euclidean_distances
from scipy.optimize import linear_sum_assignment

# Necessary to run the script from visual studio code
path = __file__
file_location_subfolders = 3 #Number of folder to go up to reach root of package
for _ in range(file_location_subfolders):
    path = os.path.dirname(path)
sys.path.insert(0, path)

from mobile_robotics_slam.Extractors.Corners.CornerExtractor import CornerExtractor
corner_extractor = CornerExtractor()

# Set the parameters of the Corner Extractor
min_corner_angle = 70
max_corner_angle = 110
max_intersecton_distance = 0.8
corner_extractor.set_corner_params(max_intersecton_distance, min_corner_angle, max_corner_angle)

# Set the parameters of the Adaptive Segment Detector
sigma_ranges = 0.3
lambda_angle = 10
merge_distance = 0.07
min_points_density = 2
min_segment_length = 0.3
corner_extractor.set_detector_params(sigma_ranges, lambda_angle, merge_distance, min_points_density, min_segment_length)

# Set the parameters of the Segment Handler
epsilon = 0.1
min_density_after_segmentation = 2
min_length_after_segmentation = 0.12
corner_extractor.set_handler_params(epsilon, min_density_after_segmentation, min_length_after_segmentation)


# Prepare the data from the reference scan
ranges = np.loadtxt(os.path.join(path, "example_scans", "scan1.txt")) # Load the ranges from the reference scan
field_of_view = 2 * np.pi # Field of view of the laser scan
angle_min = -np.pi # Minimum angle of the laser scan
angles = [angle_min + i * field_of_view / len(ranges) for i in range(len(ranges))]

start = time.time()

# corner_extractor.extract_corners(ranges, field_of_view, angle_min)
end = time.time()
# corner_extractor.plot_corners()
# extracted_corners1 = corner_extractor.get_corners()

ranges = np.loadtxt(os.path.join(path, "example_scans", "intensity_scan.txt")) # Load the ranges from the reference scan
ranges = ranges[:, 0] # Get only the ranges
corner_extractor.extract_corners(ranges, field_of_view, angle_min)
print("Time taken to extract corners: ", end-start)
corner_extractor.plot_corners()
extracted_corners2 = corner_extractor.get_corners()
