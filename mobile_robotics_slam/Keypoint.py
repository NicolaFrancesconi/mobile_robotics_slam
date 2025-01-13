import numpy as np
from itertools import chain


def flatten_list(nested_list):
            return list(chain.from_iterable(
            item if isinstance(item, list) else [item] for item in nested_list
            ))


class Keypoint:
    def __init__(self, x, y, when_stored=0.0):
        self.x_mean = x
        self.y_mean = y
        self.n_seen = 1
        self.variance_x = 0.0
        self.variance_y = 0.0
        self.covariance_xy = 0.0
        self.when_stored = when_stored

    def update(self, x, y):
        old_mean_x = self.x_mean 
        old_mean_y = self.y_mean  
        n = self.n_seen

        self.x_mean = (old_mean_x*n + x)/(n + 1) 
        self.y_mean = (old_mean_y*n + y) /(n + 1) 
        self.n_seen += 1   #increment the number of times the keypoint has been seen
        self.update_variance(old_mean_x, old_mean_y, x, y)
    
    def update_variance(self, old_mean_x, old_mean_y, x, y):
        old_variance_x = self.variance_x
        old_variance_y = self.variance_y
        n = self.n_seen
        new_mean_x = self.x_mean
        new_mean_y = self.y_mean
        
        self.variance_x = (old_variance_x*(n-1) + (x - old_mean_x)*(x - new_mean_x))/n
        self.variance_y = (old_variance_y*(n-1) + (y - old_mean_y)*(y - new_mean_y))/n
        #self.covariance_xy = (n-1)*new_covariance_xy + (x - old_mean_x)*(y - old_mean_y) # Not Correct
       

class KeypointList:
    def __init__(self):
        self.keypoints = []
        self.number_of_received_set = 0
    
    def add_keypoint(self, x, y, when_stored=0):
        self.keypoints.append(Keypoint(x, y, when_stored))

    def match_keypoints_NN(self, keypoints, threshold_match, threshold_new):
        """Match keypoints: Given the new keypoints, find the closest keypoints in the list of keypoints
        and return the indices of the matched keypoints"""
        self.number_of_received_set += 1
        matched_indices = []
        non_matched_indices = []
        ambiguous_indices = []
        idx = -1
        for keypoint in keypoints:
            idx += 1
            min_idx = -1
            min_computed_distance = np.inf
            for kp_idx, existing_keypoint in enumerate(self.keypoints):
                distance = np.sqrt((keypoint[0] - existing_keypoint.x_mean)**2 + (keypoint[1] - existing_keypoint.y_mean)**2)
                if distance < min_computed_distance and distance < threshold_match:
                    min_computed_distance = distance
                    min_idx = kp_idx
                if distance < min_computed_distance:
                    min_computed_distance = distance

            if min_idx != -1:
                matched_indices.append([idx, min_idx, min_computed_distance])
            elif min_computed_distance > threshold_new:
                non_matched_indices.append(idx)
            else:
                ambiguous_indices.append(idx)

        return matched_indices, non_matched_indices, ambiguous_indices
        
    
    def get_keypoints(self):
        return [[keypoint.x_mean, keypoint.y_mean] for keypoint in self.keypoints]
    
    def get_stable_keypoints(self, n_seen_threshold=15):
        return [[keypoint.x_mean, keypoint.y_mean] for keypoint in self.keypoints if keypoint.n_seen > n_seen_threshold]

    def get_keypoint_info(self):
        return [[keypoint.x_mean, keypoint.y_mean, keypoint.variance_x, keypoint.variance_y, keypoint.covariance_xy, keypoint.n_seen, keypoint.when_stored ] for keypoint in self.keypoints]
    
    def get_stable_keypoint_info(self, n_seen_threshold=15):
        return [[keypoint.x_mean, keypoint.y_mean, keypoint.variance_x, keypoint.variance_y, keypoint.covariance_xy ,keypoint.n_seen,  keypoint.when_stored ] for keypoint in self.keypoints if keypoint.n_seen > n_seen_threshold]