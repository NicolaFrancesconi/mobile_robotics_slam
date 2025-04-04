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

class LineSegmentPolarForm:
    def __init__(self, points):
        self.points = points
        self.angle = None
        self.distance = None
        self.endpoints = None

    def compute_polar_form(self):
        """Compute the polar form of the line from the segment"""
        points = np.array(self.points)
        x = points[:, 0]
        y = points[:, 1]

        # Construct the augmented matrix A
        A = np.vstack((x, y, np.ones(len(x)))).T

        # Compute the covariance matrix
        cov_matrix = np.dot(A.T, A)

        # Perform Eigenvalue Decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # The eigenvector corresponding to the smallest eigenvalue gives us the best-fit line
        line_params = eigenvectors[:, np.argmin(eigenvalues)]

        # Extract the line parameters (A, B, C)
        A_line = line_params[0]
        B_line = line_params[1]
        C_line = line_params[2]

        # Ensure the correct sign for the line parameters
        if C_line > 0:
            A_line, B_line, C_line = -A_line, -B_line, -C_line

        # Calculate the polar form (distance and angle)
        norm = np.sqrt(A_line**2 + B_line**2)
        self.distance = -C_line / norm
        self.angle = np.arctan2(B_line, A_line)

    def compute_endpoints(self):
        """Compute the endpoints of the line segment"""
        if self.angle is None or self.distance is None:
            raise ValueError("Line parameters not computed")
        points = np.array(self.points)
        x = points[:, 0]
        y = points[:, 1]
        
        d = x * np.cos(self.angle) + y * np.sin(self.angle) - self.distance
        x_proj = x - np.cos(self.angle) * d
        y_proj = y - np.sin(self.angle) * d
        if self.angle**2 == 0 or self.angle**2 == np.pi**2:
            
            idx_start = np.argmin(y_proj)
            idx_end = np.argmax(y_proj)

        else:
            
            idx_start = np.argmin(x_proj)
            idx_end = np.argmax(x_proj)
        
        start_point = np.array([x_proj[idx_start], y_proj[idx_start]])
        end_point = np.array([x_proj[idx_end], y_proj[idx_end]]) 
        self.endpoints = np.array([start_point, end_point])

    def compute_segment_properties(self):
        """Compute the properties of the segment"""
        self.compute_polar_form()
        self.compute_endpoints()
        

        

    def plot_fitted_line(self):
        """Plot the fitted line to a segment"""
        if self.angle is None or self.distance is None:
            print("Line parameters not computed")
            return
        x = np.array([point[0] for point in self.points])
        y = np.array([point[1] for point in self.points])
        
        x_line = np.linspace(min(x), max(x), 10)
        y_line = (-np.cos(self.angle)/np.sin(self.angle))*x_line + self.distance/np.sin(self.angle)
        plt.plot(x_line, y_line, color='red', label='Fitted line')
        plt.scatter(x, y, color='blue', s=10, label='Segment points')
        plt.axis('equal')
        plt.grid()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()


class SegmentHandler:
    def __init__(self):
        self.segments = []
        self.epsilon = None
        self.min_length_after_segmentation = None
        self.min_density_after_segmentation = None
    
    def clear_segments(self):
        self.segments = []

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_min_length_after_segmentation(self, min_length):
        self.min_length_after_segmentation = min_length

    def set_min_density_after_segmentation(self, min_density):
        self.min_density_after_segmentation = min_density

    def add_segment(self, segment):        
        self.segments.append(LineSegmentPolarForm(segment))

    def remove_segment(self, segment):
        self.segments.remove(segment)

    def remove_short_segments(self):
        if self.min_length_after_segmentation is None:
            raise ValueError("Minimum length not set")
        self.segments = [segment for segment in self.segments if np.linalg.norm(segment[0]- segment[-1]) > self.min_length_after_segmentation]

    def remove_low_density_segments(self):
        if self.min_density_after_segmentation is None:
            raise ValueError("Minimum density not set")
        self.segments = [segment for segment in self.segments if len(segment) > self.min_density_after_segmentation]
    
    def compute_segments_properties(self):
        for segment in self.segments:
            segment.compute_polar_form()
            segment.compute_endpoints()


    def plot_segments(self):
        """Plot the segments"""

        colormap = plt.cm.get_cmap("tab20", len(self.segments))
        for segment in self.segments:
            color = colormap(self.segments.index(segment))
            x = np.array([point[0] for point in segment.points])
            y = np.array([point[1] for point in segment.points])
            plt.scatter(x, y, s=5, color=color)
            plt.scatter(x[0], y[0], s=150, color=color, marker='s', edgecolors='black', zorder=3)  
            plt.scatter(x[-1], y[-1], s=150, color=color, marker='^', edgecolors='black', zorder=3)

        plt.plot(color='black', label='Segments')
        plt.scatter([], [], marker='s',s=150, color='black', label='Segments Startpoints')
        plt.scatter([], [], marker='^',s=150, color='black', label='Segments Endpoints')
        plt.title("Ramer-Douglas-Peucker Segmentation")
        plt.xlim(-8, 8)
        plt.ylim(-6, 3)
        plt.legend()
        #plt.grid()
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.show()

    def line_fit_two_points(self, first_point, last_point):
        """Fits a line from two points"""
        dx = last_point[0] - first_point[0]
        dy = last_point[1] - first_point[1]
        x1, y1 = first_point
        if np.abs(dx) <= 1e-8:
            m = 1e7*np.sign((dy)*(dx))
            b = y1 - m * x1
        else:
            m = (dy) / (dx)
            b = y1 - m * x1
        return m, b
    
    def distance_point_to_line(self, point, m, b):
        """Computes perpendicular distance from a point to a line"""
        return np.abs(m * point[0] - point[1] + b) / np.sqrt(m**2 + 1)

    
    def Ramer_Douglas_Peucker_Segmentation(self):
        """Ramer-Douglas-Peucker algorithm used for line segmentation given a cluster of points"""
        if self.epsilon is None:
            raise ValueError("Epsilon not set")
        if self.min_length_after_segmentation is None:
            raise ValueError("Minimum length not set")
        if self.min_density_after_segmentation is None:
            raise ValueError("Minimum density not set")
        epsilon = self.epsilon
        split = True
        while split:
            split = False
            for segment in self.segments:
                points = np.array(segment.points)
                # Fit a line from the first to the last point
                m, b = self.line_fit_two_points(points[0], points[-1])
                # Find the point with the maximum distance from the line
                x = points[:, 0]
                y = points[:, 1]
                denominator = np.sqrt(m**2 + 1)
                distances = np.abs(m * x - y + b) / denominator
                #distances = np.array([self.distance_point_to_line(point, m, b) for point in points])
                max_distance_idx = np.argmax(distances)
                max_distance = distances[max_distance_idx]
                # If the point is greater than epsilon, split the segment at the point
                if max_distance > epsilon:
                    segment1 = points[:max_distance_idx]
                    segment2 = points[max_distance_idx:]
                    segment1_density = len(segment1)
                    segment2_density = len(segment2)
                    segment1_length = np.linalg.norm(segment1[0] - segment1[-1])
                    segment2_length = np.linalg.norm(segment2[0] - segment2[-1])
                    split = True

                    if segment1_density > self.min_density_after_segmentation and segment1_length > self.min_length_after_segmentation:
                        self.add_segment(segment1)
                    if segment2_density > self.min_density_after_segmentation and segment2_length > self.min_length_after_segmentation:
                        self.add_segment(segment2)
                    self.remove_segment(segment)

    def merge_similar_lines_segments(self, angle_threshold, distance_threshold, close_endpoints_threshold):
        """Merge similar lines segments"""

        merge = True
        while merge:
            merge = False
            segments = self.segments.copy()
            for i in range(len(segments)):
                segment = segments[i]
                for j in range(i+1, len(segments)):
                    other_segment = segments[j]
                    angle_error = np.abs((segment.angle - other_segment.angle + np.pi) % (2 * np.pi) - np.pi)
                    distance_error = np.abs(segment.distance - other_segment.distance)
                    end_point_min_distance = min(np.linalg.norm(segment.endpoints[0] - other_segment.endpoints[0]), np.linalg.norm(segment.endpoints[0] - other_segment.endpoints[1]), np.linalg.norm(segment.endpoints[1] - other_segment.endpoints[0]), np.linalg.norm(segment.endpoints[1] - other_segment.endpoints[1]))
                    if angle_error < angle_threshold and distance_error < distance_threshold and end_point_min_distance < close_endpoints_threshold:
                        merged_segment = [point for point in segment.points]
                        merged_segment.extend([point for point in other_segment.points])
                        self.add_segment(merged_segment)
                        self.segments[-1].compute_segment_properties()
                        self.remove_segment(segment)
                        self.remove_segment(other_segment)
                        merge = True
                        break
                if merge:
                    break


        

    


        

    