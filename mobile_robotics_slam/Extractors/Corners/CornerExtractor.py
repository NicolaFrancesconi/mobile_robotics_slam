import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

path = __file__
file_location_subfolders = 4 #Number of folder to go up to reach root of package
for _ in range(file_location_subfolders):
    path = os.path.dirname(path)
sys.path.insert(0, path)

from mobile_robotics_slam.Extractors.Corners.SegmentDetector import SegmentDetector
from mobile_robotics_slam.Extractors.Corners.SegmentHandler import SegmentHandler, LineSegmentPolarForm

class Corner:
    def __init__(self, x: float, y: float, angle: float, segment1: LineSegmentPolarForm, segment2: LineSegmentPolarForm):
        self.x = x
        self.y = y
        self.angle = angle
        self.segment1 = segment1
        self.segment2 = segment2
        self.orientation = self.compute_orientation()
        self.orientation_quaternion = self.orientation_to_quaternion(self.orientation)
        #self.descriptor = self.BinaryShapeContextDescriptor(1, 16, 2)
        

    def get_position(self):
        return np.array([self.x, self.y])

    def compute_orientation(self):
        """Compute the orientation of the corner"""
        
        origin = np.array([self.x, self.y])
        endpoints1 = np.array(self.segment1.endpoints)
        endpoints2 = np.array(self.segment2.endpoints)
        if np.linalg.norm(endpoints1[0] - origin) < np.linalg.norm(endpoints1[1] - origin):
            endpoint1 = endpoints1[1]
            direction1 = (endpoint1 - origin)/np.linalg.norm(endpoint1 - origin)
        else:
            endpoint1 = endpoints1[0]
            direction1 = (endpoint1 - origin)/np.linalg.norm(endpoint1 - origin)
        
        if np.linalg.norm(endpoints2[0] - origin) < np.linalg.norm(endpoints2[1] - origin):
            endpoint2 = endpoints2[1]
            direction2 = (endpoint2 - origin)/np.linalg.norm(endpoint2 - origin)    
        else:
            endpoint2 = endpoints2[0]
            direction2 = (endpoint2 - origin)/np.linalg.norm(endpoint2 - origin)

        bisector = (direction1 + direction2)/np.linalg.norm(direction1 + direction2)
        
        return bisector

        
    def orientation_to_quaternion(self, orientation):
        """Convert the orientation vector to a quaternion"""
        theta = np.arctan2(orientation[1], orientation[0])
        x, y = 0.0, 0.0
        z, w = np.sin(theta/2), np.cos(theta/2)
        return np.array([ x, y, z, w])

class CornerExtractor:
    def __init__(self):
        self.segment_detector = SegmentDetector()
        self.segment_handler = SegmentHandler()
        self.corners = None
        self.max_intersecton_distance = None
        self.min_corner_angle = None
        self.max_corner_angle = None
        self.max_extraction_range = 4
        self.image_cnt = 0
    def set_corner_params(self, max_intersecton_distance, min_corner_angle, max_corner_angle):
        """Set the parameter of the Corner Extractor
        Input: 
            max_intersecton_distance: Float in Meters
            min_corner_angle: Float in Degrees
            max_corner_angle: Float in Degrees
        """
        self.max_intersecton_distance = max_intersecton_distance
        self.min_corner_angle = np.deg2rad(min_corner_angle)
        self.max_corner_angle = np.deg2rad(max_corner_angle)
    
    def set_detector_params(self, sigma_ranges, lambda_angle, merge_distance, min_points_density, min_segment_length):
        """Set the parameter of the Adaptive Segment Detector
        Input: 
            sigma_ranges: Float in Meters 
            lambda_angle: Float in Degrees
        """
        self.segment_detector.set_lambda_angle(lambda_angle)
        self.segment_detector.set_sigma_ranges(sigma_ranges)
        self.segment_detector.set_merge_distance(merge_distance)
        self.segment_detector.set_min_points_density(min_points_density)
        self.segment_detector.set_min_segment_length(min_segment_length)

    def set_handler_params(self, epsilon, min_density_after_segmentation, min_length_after_segmentation):
        """Set the parameter of the Segment Handler
        Input: 
            epsilon: Float in Meters
            min_density_after_segmentation: Int
            min_length_after_segmentation: Float in Meters
        """
        self.segment_handler.set_epsilon(epsilon)
        self.segment_handler.set_min_density_after_segmentation(min_density_after_segmentation)
        self.segment_handler.set_min_length_after_segmentation(min_length_after_segmentation)

    def get_corners(self):
        if self.corners is None:
            raise ValueError("Corners not computed")
        return self.corners
    

    def find_corners(self):
        """Find the corners by intersecting the lines of the segments"""
        if self.max_intersecton_distance is None or self.min_corner_angle is None or self.max_corner_angle is None:
            raise ValueError("Corner Extractor Parameters not set")
        min_angle = self.min_corner_angle
        max_angle = self.max_corner_angle
        max_distance = self.max_intersecton_distance
        corners = []
        segments = self.segment_handler.segments

        for i, segment in enumerate(segments):
            for j in range(i + 1, len(segments)):  # Avoiding redundant comparisons
                other_segment = segments[j]
                # Angle error computation
                angle_error = np.abs((segment.angle - other_segment.angle + np.pi) % (2 * np.pi) - np.pi)
                
                # If the angle error is within the specified range
                if min_angle < angle_error < max_angle:
                    intersection = self.intersection_of_lines(segment.distance, segment.angle, other_segment.distance, other_segment.angle)
                    
                    endpoints1 = segment.endpoints
                    endpoints2 = other_segment.endpoints


                    x1min = min(endpoints1[0][0], endpoints1[1][0])
                    x1max = max(endpoints1[0][0], endpoints1[1][0])
                    y1min = min(endpoints1[0][1], endpoints1[1][1])
                    y1max = max(endpoints1[0][1], endpoints1[1][1])

                    x2min = min(endpoints2[0][0], endpoints2[1][0])
                    x2max = max(endpoints2[0][0], endpoints2[1][0])
                    y2min = min(endpoints2[0][1], endpoints2[1][1])
                    y2max = max(endpoints2[0][1], endpoints2[1][1])
                    
                    x_intersection, y_intersection = intersection

                    #Check if the intersection point is within the segment
                    if (x1min <= x_intersection <= x1max and
                        y1min <= y_intersection <= y1max):
                        if (np.linalg.norm(intersection - endpoints2[0]) < max_distance or
                            np.linalg.norm(intersection - endpoints2[1]) < max_distance):
                            corners.append(Corner(x_intersection, y_intersection, angle_error, segment, other_segment))
                    elif (x2min <= x_intersection <= x2max and
                            y2min <= y_intersection <= y2max):
                        if (np.linalg.norm(intersection - endpoints1[0]) < max_distance or
                            np.linalg.norm(intersection - endpoints1[1]) < max_distance):
                            corners.append(Corner(x_intersection, y_intersection, angle_error, segment, other_segment))
                    elif ((np.linalg.norm(intersection - endpoints2[0]) < max_distance or
                            np.linalg.norm(intersection - endpoints2[1]) < max_distance) and
                            (np.linalg.norm(intersection - endpoints1[0]) < max_distance or
                            np.linalg.norm(intersection - endpoints1[1]) < max_distance)):
                            corners.append(Corner(x_intersection, y_intersection, angle_error, segment, other_segment))
        
        self.corners = corners
        
    
    def intersection_of_lines(self, d1, angle1, d2, angle2):
        cos1, sin1 = np.cos(angle1), np.sin(angle1)
        cos2, sin2 = np.cos(angle2), np.sin(angle2)
        # Using direct formula to solve the system of equations without constructing the full matrix
        det = cos1 * sin2 - sin1 * cos2
        x = (d1 * sin2 - d2 * sin1) / det
        y = (d2 * cos1 - d1 * cos2) / det
        return np.array([x, y])
    
    def distance_to_intersection_point(self, segment, intersection):
        squared_distances = np.sum((np.array(segment.points) - np.array(intersection))**2, axis=1)
        return np.sqrt(np.min(squared_distances))

    def extract_corners(self, scan_ranges, field_of_view, min_angle, robot_pose=np.array([0, 0, 0])):
        self.segment_detector.detect_segments(scan_ranges, field_of_view, min_angle, robot_pose)
         # Merge Close Segments That were splitted by outliers
        self.segment_handler.clear_segments()
        for segment in self.segment_detector.get_segments():
            self.segment_handler.add_segment(segment)

        self.segment_handler.Ramer_Douglas_Peucker_Segmentation()
        self.segment_handler.compute_segments_properties()
        #self.segment_handler.merge_similar_lines_segments(np.deg2rad(10), 0.08, 1)
        self.find_corners()

    def plot_corners(self, path=os.path.dirname(__file__), ranges=None):
        if self.corners is None:    
            raise ValueError("Corners not computed")
        plt.figure()
        plt.title("Segments and Extracted Corners")
        x_seg = []
        y_seg = []
        for i, segment in enumerate(self.segment_handler.segments):
            x= [point[0] for point in segment.points]
            y= [point[1] for point in segment.points]
            plt.plot(x, y,  label = "Segment"+str(i))

        if ranges is not None:
            angles = np.linspace(-np.pi, np.pi, len(ranges))
            x = ranges* np.cos(angles)
            y= ranges* np.sin(angles)
            plt.scatter(x, y, color='yellow', s=1, label = "LaserScan")


        x_corn  = []
        y_corn = []
        for corner in self.corners:
            x_corn.append(corner.x)
            y_corn.append(corner.y)
            #plt.quiver(corner.x, corner.y, corner.orientation[0], corner.orientation[1], color=['r'], scale=5)

        plt.scatter(x_corn, y_corn, color='red', s=10, label = "Corners")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.savefig(os.path.join(path, f"Corners{self.image_cnt}.png"))
        self.image_cnt += 1
        plt.legend()
        plt.show() 



        

        

        
        


        

    