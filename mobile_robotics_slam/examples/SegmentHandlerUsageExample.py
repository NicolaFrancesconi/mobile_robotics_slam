import os
import sys
import numpy as np

path = __file__
file_location_subfolders = 3 #Number of folder to go up to reach root of package
for _ in range(file_location_subfolders):
    path = os.path.dirname(path)
sys.path.insert(0, path)

from mobile_robotics_slam.Extractors.Corners.SegmentHandler import SegmentHandler
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

# Run the detector: Detect Breakpoints and Generate Segments
detector.detect_segments(ranges, field_of_view, angle_min)

## Test Segments for the Segment Handler

# Segment 1
def generate_corner_segment(num_points1=100, num_points2=100, angle_degrees=45, length1=0.5, length2=0.4):
    """
    Generates points representing two line segments that meet at a corner, with a specified number of points.

    Parameters:
        num_points1 (int): Number of points in the first segment.
        num_points2 (int): Number of points in the second segment.
        angle_degrees (float): Angle between the two segments in degrees.
        length1 (float): Length of the first segment.
        length2 (float): Length of the second segment.

    Returns:
        np.ndarray: Array of points forming a corner.
    """
    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)

    # First segment (along x-axis), equally spaced points
    segment1_x = np.linspace(0, length1, num_points1)
    segment1_y = np.zeros(num_points1)
    segment1 = np.vstack((segment1_x, segment1_y)).T

    # Second segment (rotated by the given angle), equally spaced points
    segment2_x = length1 + np.linspace(0, length2 * np.cos(angle_radians), num_points2)
    segment2_y = np.linspace(0, length2 * np.sin(angle_radians), num_points2)
    segment2 = np.vstack((segment2_x, segment2_y)).T

    # Combine both segments
    corner_segment = np.vstack((segment1, segment2))
    
    return corner_segment

# Generate the corner segment
segments = []
segments.append(generate_corner_segment())

segments = detector.segments

# Create a segment handler
segment_handler = SegmentHandler()

for segment in segments:
    segment_handler.add_segment(segment) # Add the segments to the segment handler

segment_handler.plot_segments() # Show the segments before the Ramer-Douglas-Peucker algorithm
segment_handler.set_epsilon(0.1) # Set the epsilon parameter for the Ramer-Douglas-Peucker algorithm
segment_handler.set_min_density_after_segmentation(8) # Set the minimum points density parameter for the Ramer-Douglas-Peucker algorithm
segment_handler.set_min_length_after_segmentation(0.15) # Set the minimum length after segmentation parameter for the Ramer-Douglas-Peucker algorithm
segment_handler.Ramer_Douglas_Peucker_Segmentation() # Run the Ramer-Douglas-Peucker algorithm
segment_handler.plot_segments() # Show the segments after the Ramer-Douglas-Peucker algorithm


# for segment in segment_handler.segments:
#     segment.compute_polar_form() # Compute the polar form for each segment
#     segment.plot_fitted_line() # Plot the fitted line for each

