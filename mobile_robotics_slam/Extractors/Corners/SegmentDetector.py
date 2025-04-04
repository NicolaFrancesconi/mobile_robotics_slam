import numpy as np
import matplotlib.pyplot as plt

class SegmentDetector:
    def __init__(self):
        """ This class is used to detect breakpoints in a 2D point cloud
        generating pairs of points that represent the initial and final points of a segment
        and collect the points of the segments in a list of segments"""
        self.breakpoints_idx = None
        self.sigma_ranges = None
        self.lambda_angle = None
        self.segments = None
        self.ranges = None
        self.angles = None
        self.merge_distance = None
        self.min_points_density = None
        self.min_segment_length = None

    def set_merge_distance(self, merge_distance):
        """Sets the merge distance threshold"""
        self.merge_distance = merge_distance

    def set_min_points_density(self, min_points_density):
        """Sets the minimum points density"""
        self.min_points_density = min_points_density

    def set_min_segment_length(self, min_segment_length):
        """Sets the minimum segment length"""
        self.min_segment_length = min_segment_length

    def set_sigma_ranges(self, sigma_ranges):
        """Sets the sigma ranges """
        self.sigma_ranges = sigma_ranges

    def set_lambda_angle(self, lambda_angle):
        """Sets the lambda angle in degrees"""
        self.lambda_angle = np.deg2rad(lambda_angle)

    def get_breakpoints_idx(self):
        """Returns the detected breakpoints"""
        if self.breakpoints_idx is None:
            raise ValueError("No breakpoints detected. Make sure breakpoints are identified before calling this function.")
        return self.breakpoints_idx.copy()
    
    def get_segments(self):
        """Returns the detected segments"""
        if self.segments is None:
            raise ValueError("No segments detected. Make sure segments are identified before calling this function.")
        return self.segments

    def polar_to_cartesian(self, range, angle):
        """Converts polar coordinates to cartesian"""
        x = range * np.cos(angle)
        y = range * np.sin(angle)
        return np.array([x, y])


    def detect_segments(self, scan_ranges, field_of_view, angle_min, robot_pose=np.array([0, 0, 0])):
        """Detects breakpoints in the scan"""
        
        if self.sigma_ranges is None or self.lambda_angle is None:
            raise ValueError("Sigma Ranges and Lambda Angle should be set before detecting breakpoints")

        ranges = np.array(scan_ranges)
        angle_resolution = field_of_view / len(ranges)
        angles = np.linspace(angle_min, angle_min + field_of_view, len(ranges))

        # Remove Ranges with Inf Values and their corresponding angles
        ranges[np.isinf(ranges)] = 0

        self.ranges = ranges
        self.angles = angles

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        positions = np.array([x, y]).T

        # Precompute Cartesian coordinates for all ranges
        #positions = np.array([self.polar_to_cartesian(r, a) for r, a in zip(ranges, angles)])

        # Compute differences between consecutive points in Cartesian space
        delta_positions = positions[1:] - positions[:-1]
        euclidean_distances = np.linalg.norm(delta_positions, axis=1)

        # Compute angle differences (delta_phi) and adaptive thresholds
        delta_phi = np.diff(angles)  # angle differences between consecutive points
        D_max = (ranges[:-1] * np.sin(delta_phi)) / np.sin(self.lambda_angle - delta_phi) + 3 * self.sigma_ranges

        # Initialize variables for breakpoints detection
        first_discontinuity_detected = False
        break_points_pairs = []
        starting_breakpoint = 0

        for idx, (dist, d_max) in enumerate(zip(euclidean_distances, D_max)):
            if dist > d_max:
                if not first_discontinuity_detected:
                    first_discontinuity_detected = True
                    starting_breakpoint = idx + 1  # Adjust for 0-based indexing
                else:
                    break_points_pairs.append((starting_breakpoint, idx))
                    starting_breakpoint = idx + 1  # Update starting point for next segment

        # Filter out pairs with less than 2 points
        break_points_pairs = [pair for pair in break_points_pairs if (pair[1] - pair[0]) > 1]

        # Handle the circular nature of the scan and append the last breakpoint pair if needed
        if len(break_points_pairs) > 0:
            delta_idx = (len(ranges) - (break_points_pairs[-1][1] + 1)) + break_points_pairs[0][0]
            if delta_idx > 0:
                break_points_pairs.append((break_points_pairs[-1][1] + 1, break_points_pairs[0][0] - 1))

        else: # No breakpoints detected
            break_points_pairs.append((0, len(ranges) - 1))

        

        self.breakpoints_idx = break_points_pairs
        self.generate_segments(ranges, angles, robot_pose)
        #self.merge_close_segments()
        self.remove_low_density_segments()
        self.remove_short_segments()


    def generate_segments(self, ranges, angles, robot_pose):
        """Given the ranges and angles and the breakpoints indices, generates the segments"""
        if self.breakpoints_idx is None:
            raise ValueError("Breakpoints should be detected before generating segments")
        segments = []

        for (start_idx, end_idx) in self.breakpoints_idx:
            if start_idx < end_idx:
                # Direct slicing for non-circular segment
                segment_ranges = ranges[start_idx:end_idx + 1]
                segment_angles = angles[start_idx:end_idx + 1]
            else:
                # Circular segment
                segment_ranges = np.concatenate((ranges[start_idx:], ranges[:end_idx + 1]))
                segment_angles = np.concatenate((angles[start_idx:], angles[:end_idx + 1]))
            
            # Vectorized Cartesian conversion for the entire segment
            x = segment_ranges * np.cos(segment_angles + robot_pose[2] ) + robot_pose[0]
            y = segment_ranges * np.sin(segment_angles+ robot_pose[2]) + robot_pose[1]
            segment_positions = np.array([x, y]).T

            segments.append(segment_positions)
        self.segments = segments

    def merge_close_segments(self):
        """Merges the segments that have start and end points close to each other"""
        if self.merge_distance is None:
            raise ValueError("Merge Distance should be set")
        distance_threshold = self.merge_distance
        if self.segments == None:
            raise ValueError("Segments should be detected before merging")
        merged_segments = self.segments.copy()
        merged = True
        start_idx = 0
        while merged:
            merged = False
            for i in range( start_idx , len(merged_segments)-1):
                for j in range(i+1, len(merged_segments)):
                    min_distance = self.min_endpoints_distance(merged_segments[i], merged_segments[j], 1)
                    if min_distance < distance_threshold:
                        new_segment = merged_segments[i] + merged_segments[j]
                        merged_segments.pop(j)
                        merged_segments.pop(i)
                        merged_segments.insert(i, new_segment)
                        start_idx = i
                        merged = True
                        break
                if merged:
                    break
        self.segments = merged_segments


    def min_endpoints_distance(self, segment1, segment2, num_points):
        # Extract endpoint points from each segment
        segment1_points = np.array(segment1[:num_points] + segment1[-num_points:])
        segment2_points = np.array(segment2[:num_points] + segment2[-num_points:])
        distances = np.linalg.norm(segment1_points[:, np.newaxis, :] - segment2_points[np.newaxis, :, :], axis=2)
        return np.min(distances)

    
    def remove_short_segments(self):
        """Removes the segments that have less than min_length points"""
        if self.min_segment_length is None:
            raise ValueError("Minimum Segment Length Has to be Set")
        min_distance = self.min_segment_length
        if self.segments is None:
            raise ValueError("Segments should be detected before removing short segments")
        self.segments = [segment for segment in self.segments if np.linalg.norm(segment[0] - segment[-1]) > min_distance]

        
    def remove_low_density_segments(self):
        """Removes the segments that have less than min_density points"""
        if self.min_points_density is None:
            raise ValueError("Minimum Points Density Has to be Set")
        min_density = self.min_points_density
        if self.segments is None:
            raise ValueError("Segments should be detected before removing low density segments")
        original_length = len(self.segments)
        self.segments = [segment for segment in self.segments if len(segment) > min_density]         

    def plot_segments_and_scan(self):
        """Plots the segments"""
        if self.segments is None:
            raise ValueError("Segments should be detected before plotting")
        if self.ranges is None or self.angles is None:
            raise ValueError("Laser Scan Data should be provided before plotting")
        
        x = self.ranges * np.cos(self.angles)
        y = self.ranges * np.sin(self.angles)
        plt.scatter(x, y, s=5, label='Laser Scan Data', color='blue')
        colormap = plt.cm.get_cmap("tab20", len(self.segments))

        for i, segment in enumerate(self.segments):  
            x = [point[0] for point in segment]  
            y = [point[1] for point in segment]  
            color = colormap(i)  # Generate a random color  

            plt.plot(x, y, color=color, linewidth=3, zorder=1)  
            plt.scatter(x[0], y[0], s=150, color=color, marker='s', edgecolors='black', zorder=3)  
            plt.scatter(x[-1], y[-1], s=150, color=color, marker='^', edgecolors='black', zorder=3)  

        plt.scatter([], [], marker='s', s=150, color='black', label='Segments Startpoints')
        plt.scatter([], [], marker='^', s=150, color='black', label='Segments Endpoints')

        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.legend()
        plt.title("Adaptive Breakpoint Detector Segmentation")
        plt.xlim(-8, 8)
        plt.ylim(-6, 3)
        plt.show()

    def plot_detected_breakpoints(self):
        """Plots the detected breakpoints"""
        if self.breakpoints_idx is None:
            raise ValueError("Breakpoints should be detected before plotting")
        if self.ranges is None or self.angles is None:
            raise ValueError("Laser Scan Data should be provided before plotting")
        
        x = self.ranges * np.cos(self.angles)
        y = self.ranges * np.sin(self.angles)
        plt.scatter(x, y, s=0.2, label='Laser Scan Data', color='yellow')
        for i, (start_idx, end_idx) in enumerate(self.breakpoints_idx):
            x = [self.ranges[start_idx] * np.cos(self.angles[start_idx]), self.ranges[end_idx] * np.cos(self.angles[end_idx])]
            y = [self.ranges[start_idx] * np.sin(self.angles[start_idx]), self.ranges[end_idx] * np.sin(self.angles[end_idx])]
            plt.scatter(x, y, s=3,  label='Breakpoint_Pair_'+str(i))

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.title("Detected Pair of Breakpoints in Synthetic Laser Scan Data")
        plt.axis("equal")
        plt.show()




