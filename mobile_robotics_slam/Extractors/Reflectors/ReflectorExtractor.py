import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.cluster import DBSCAN
from scipy.optimize import least_squares

class Reflector:
    def __init__(self, x_center, y_center, points, radius):
        self.x = x_center
        self.y = y_center
        self.radius = radius
        self.points = points
    
    def get_position(self):
        return np.array([self.x, self.y])
        
class ReflectorExtractor:
    def __init__(self):
        self.reflector_list = []
        self.scan_points = []
        self.scan_intensities = []
        self.reflector_clusters = []
        self.min_reflector_points = 4
        self.min_reflector_radius = 0.02
        self.max_reflector_radius = 0.15
        self.min_reflector_intensity = 1000
        self.cluster_radius = 0.8
        self.max_range_extraction = 30
        self.min_range_extraction = 0.01

    def set_min_reflector_points(self, min_reflector_points):
        self.min_reflector_points = min_reflector_points
    def set_min_reflector_radius(self, min_reflector_radius):
        self.min_reflector_radius = min_reflector_radius
    def set_max_reflector_radius(self, max_reflector_radius):
        self.max_reflector_radius = max_reflector_radius
    def set_min_reflector_intensity(self, min_reflector_intensity):
        self.min_reflector_intensity = min_reflector_intensity
    def set_cluster_radius(self, cluster_radius):
        self.cluster_radius = cluster_radius
    def set_max_range_extraction(self, max_range_extraction):
        self.max_range_extraction = max_range_extraction
    def set_min_range_extraction(self, min_range_extraction):
        self.min_range_extraction = min_range_extraction
    


    def extract_reflectors(self, scan_ranges, scan_intensities, field_of_view, angle_min, robot_pose=np.array([0, 0, 0])):
        intensities = np.array(scan_intensities)
        scan_angles = np.linspace(angle_min, angle_min + field_of_view, len(scan_ranges))

        self.scan_points = np.vstack((scan_ranges*np.cos(scan_angles+ robot_pose[2])+ robot_pose[0], scan_ranges*np.sin(scan_angles+ robot_pose[2])+ robot_pose[1])).T
        self.scan_intensities = np.copy(intensities)
        ranges = scan_ranges[intensities > self.min_reflector_intensity]
        angles = scan_angles[intensities > self.min_reflector_intensity]

        
        
        angles = angles[(ranges > self.min_range_extraction) & (ranges < self.max_range_extraction)]
        ranges = ranges[(ranges > self.min_range_extraction) & (ranges < self.max_range_extraction)]

        if len(ranges) < self.min_reflector_points:
            return
                
        x = ranges * np.cos(angles + robot_pose[2]) + robot_pose[0]
        y = ranges * np.sin(angles + robot_pose[2]) + robot_pose[1]
        points = np.vstack((x, y)).T 

        self.cluster_points(points)
        self.extract_reflectors_from_clusters()

    def cluster_points(self, points):
        # Group points into clusters given a maximum distance bewteen points that belong to the same cluster
        clusters = []
        db = DBSCAN(eps=self.cluster_radius, min_samples=self.min_reflector_points)  # Adjust eps and min_samples as needed
        labels = db.fit_predict(points)

        # Group points into clusters
        clusters = []
        for label in np.unique(labels):
            if label != -1:  # -1 indicates noise points
                cluster_points = points[labels == label]
                clusters.append(cluster_points)
        self.reflector_clusters = clusters

        #Plot clusters
        for cluster in clusters:
            plt.scatter(cluster[:, 0], cluster[:, 1], s=9)
        
        plt.xlim(-8, 8)
        plt.ylim(-6, 3)
        plt.title('Clusters of High Intensity Points With DBSCAN')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        #plt.legend()
        plt.show()
        return clusters
        

    def extract_reflectors_from_clusters(self):
        reflectors = []
        for cluster in self.reflector_clusters:
            x_center, y_center, radius = self.fit_circle(cluster)
            # # Plot the Points and the Circle
            # plt.scatter(cluster[:, 0], cluster[:, 1], c='red', s=30, label='Cluster Points') # type: ignore
            # plt.scatter(x_center, y_center, c='black', s=20, label='Fitted Circle Center') # type: ignore
            # circle = plt.Circle((x_center, y_center), radius, color='black', fill=False) # type: ignore
            # plt.gca().add_artist(circle)
            # plt.xlim(x_center- 0.2, x_center + 0.2)
            # plt.ylim(y_center - 0.2, y_center + 0.2)
            # plt.gca().set_aspect('equal', adjustable='datalim')
            # plt.title('Cluster Points and Fitted Circle')
            # plt.xlabel('X [m]')
            # plt.ylabel('Y [m]')
            # plt.legend()
            # plt.show()
            if radius > self.min_reflector_radius and radius < self.max_reflector_radius:
                reflectors.append(Reflector(x_center, y_center, cluster, radius))
        self.reflector_list = reflectors


    def fit_circle(self, points):
        x = points[:, 0]
        y = points[:, 1]
        # Initial guess for circle parameters (cx, cy, r)
        x_m, y_m = np.mean(x), np.mean(y)
        initial_guess = [x_m, y_m, np.mean(np.sqrt((x - x_m)**2 + (y - y_m)**2))]
        # Define the residuals for least-squares optimization
        def residuals(params):
            cx, cy, r = params
            return np.sqrt((x - cx)**2 + (y - cy)**2) - r
        # Optimize to find best-fit circle parameters
        result = least_squares(residuals, initial_guess)
        cx, cy, r = result.x
        return cx, cy, r
    
    def get_reflectors(self):
        return self.reflector_list
    
    def plot_reflectors(self):
        # Plot Reflectors Fitted Circles and Scan Points
        plt.figure()
        HI_mask = self.scan_intensities >= self.min_reflector_intensity
        LI_mask = self.scan_intensities < self.min_reflector_intensity
        plt.scatter(self.scan_points[LI_mask, 0], self.scan_points[LI_mask, 1], c='b', s=5, label='Low Intensity Points') # type: ignore
        plt.scatter(self.scan_points[HI_mask, 0], self.scan_points[HI_mask, 1], c='red', s=5, label='High Intensity Points') # type: ignore
        for i, reflector in enumerate(self.reflector_list):
            plt.scatter(reflector.x, reflector.y, c='orange', s=160, edgecolors='black', alpha=0.9)
            # circle = plt.Circle((reflector.x, reflector.y), reflector.radius, color='black', fill=False) # type: ignore
            # plt.gca().add_artist(circle)

        # #Add to legend the reflector circles
        plt.scatter([], [], c='orange', s=160, edgecolors='black', alpha=0.9, label='Reflector')
        plt.xlim(-8, 8)
        plt.ylim(-6, 3)
        plt.title('Point Cloud with Intensity')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.legend()
        plt.show()


    

        





        
        


        

    