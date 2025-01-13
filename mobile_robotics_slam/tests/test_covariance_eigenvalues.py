import os
import sys
import numpy as np

import matplotlib.pyplot as plt

# adding localization_lib to the system path
sys.path.insert(
    0, os.path.join(os.getcwd(), "src", "application", "mobile_robotics_slam")
)

from mobile_robotics_slam.Keypoint import KeypointList, Keypoint

# Read the data from the file where each line is a keypoint with the format: x y variance_x variance_y covariance_xy n_seen when_stored
def read_keypoints_from_file(file_path):
        kepoints = KeypointList()

        with open(file_path, "r") as f:
            for line in f:
                data = line.split(" ")
                x = float(data[0])
                y = float(data[1])
                variance_x = float(data[2])
                variance_y = float(data[3])
                covariance_xy = float(data[4])
                n_seen = int(data[5])
                when_stored = int(data[6])
                keypoint = Keypoint(x, y, when_stored)
                keypoint.variance_x = variance_x
                keypoint.variance_y = variance_y
                keypoint.covariance_xy = covariance_xy
                keypoint.n_seen = n_seen

                kepoints.keypoints.append(keypoint)

        return kepoints

def get_keypoints_squared_eigenvalues(keypoint_list, n_seen_min):
    """
    Return the squared eigenvalues of the covariance matrix of the keypoints in the keypoint_list of
    keypoints that have been seen at least n_seen_min times
    """
    keypoint_covariance = [np.array([[keypoint.variance_x, keypoint.covariance_xy], [keypoint.covariance_xy, keypoint.variance_y]]) for keypoint in keypoint_list.keypoints if keypoint.n_seen > n_seen_min]
    eigenvalues = [np.linalg.eigvals(covariance) for covariance in keypoint_covariance]

    squared_eigenvalues_list = []
    for eig in eigenvalues:
        squared_eigenvalues_list.append(np.sqrt(eig[0]))
        squared_eigenvalues_list.append(np.sqrt(eig[1]))
    return squared_eigenvalues_list

def compute_max_and_geometric_mean_of_squared_eigenvalues(squared_eigenvalues):
    max_squared_eigenvalue = max(squared_eigenvalues)
    #geometric_mean = np.prod(squared_eigenvalues)**(1/len(squared_eigenvalues))
    standad_mean = np.mean(squared_eigenvalues)
    return max_squared_eigenvalue, standad_mean

def compute_statistics(keypoints):
    total_keypoints = len(keypoints.keypoints)
    single_seen_keypoints = len([keypoint for keypoint in keypoints.keypoints if keypoint.n_seen == 1])
    squared_eigenvalues = get_keypoints_squared_eigenvalues(keypoints, N_SEEN_MIN)
    stable_keypoints = len([keypoint for keypoint in keypoints.keypoints if keypoint.n_seen > N_SEEN_MIN])
    max_lambda, mean_lambda = compute_max_and_geometric_mean_of_squared_eigenvalues(squared_eigenvalues)
    return total_keypoints, max_lambda, mean_lambda, single_seen_keypoints, stable_keypoints

def print_statistics(keypoints, name):
    total_keypoints, max_lambda, mean_lambda, single_seen_keypoints, stable_keypoints = compute_statistics(keypoints)
    print(f"{name}: Num Extracted Point: {total_keypoints} Max Lambda: {max_lambda:.5f} Lambda Mean: {mean_lambda:.5f} Single Point: {single_seen_keypoints} Stable Points: {stable_keypoints}")
    
print("\n\n")
print( "\t"*3,"#"*32)
print( "\t"*3 ,"Stability Analysis of Keypoints ")
print( "\t"*3,"#"*32)



N_SEEN_MIN = 30

falko_keypoints = read_keypoints_from_file("falko_keypoints.txt")
print_statistics(falko_keypoints, "FALKO")

oc_keypoints = read_keypoints_from_file("oc_keypoints.txt")
print_statistics(oc_keypoints, "OC")

my_keypoints = read_keypoints_from_file("my_keypoints.txt")
print_statistics(my_keypoints, "MY")
