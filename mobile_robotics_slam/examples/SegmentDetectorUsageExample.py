import os
import sys
import numpy as np

path = __file__
file_location_subfolders = 3 #Number of folder to go up to reach root of package
for _ in range(file_location_subfolders):
    path = os.path.dirname(path)
sys.path.insert(0, path)

from mobile_robotics_slam.Extractors.Corners.SegmentDetector import SegmentDetector

########################################################################################
##Test the BreakPointDetector
########################################################################################

# Create an instance of the BreakPointDetector
detector = SegmentDetector()

# Set parameters for the detector
sigma_ranges = 0.3
lambda_angle = 10
merge_distance_threshold = 0.07
min_points_density = 0
min_segment_length = 0.30

detector.set_sigma_ranges(sigma_ranges)
detector.set_lambda_angle(lambda_angle)
detector.set_merge_distance(0.07)
detector.set_min_points_density(0)
detector.set_min_segment_length(0.30)


#Prepare the Laser data
ranges = np.loadtxt(os.path.join(path, "example_scans", "intensity_scan.txt")) # Load the ranges from the reference scan
ranges = ranges[:, 0] # Get only the ranges
field_of_view = 2 * np.pi # Field of view of the laser scan
angle_min = -np.pi # Minimum angle of the laser scan

angles = [angle_min + i * field_of_view / len(ranges) for i in range(len(ranges))]

#Plot the laser scan
import matplotlib.pyplot as plt
x = ranges * np.cos(angles)
y = ranges * np.sin(angles)
plt.scatter(x, y, s=5, color='blue', label="Laser Scan")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("Laser Scan")
plt.legend()
plt.xlim(-8, 8)
plt.ylim(-6, 3)
plt.show()
# Run the detector: Detect Breakpoints and Generate Segments
detector.detect_segments(ranges, field_of_view, angle_min)
print("Number of segments detected: ", len(detector.segments))
detector.plot_segments_and_scan()