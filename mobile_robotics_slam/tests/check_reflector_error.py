import numpy as np
import matplotlib.pyplot as plt
import os

path = __file__
file_location_subfolders = 3 #Number of folder to go up to reach root of package
for _ in range(file_location_subfolders):
    path = os.path.dirname(path)

def read_reflectors_from_file(file_path):
    reflectors = []
    with open(file_path, "r") as f:
        for line in f:
            # remove tab and newline characters
            lin  = line.replace("\t", " ")
            data = lin.split(" ")
            x = float(data[0])
            y = float(data[1])
            position = np.array([x, y])
            reflectors.append(position)
    return reflectors

map_reflectors = read_reflectors_from_file(os.path.join(path,"example_scans", "SimulationReflectorTrueMap.txt"))


def match_landmarks_NN(self, landmarks, landmarks_map,  threshold_match, threshold_new):
        """Match keypoints: Given the new keypoints, find the closest keypoints in the list of keypoints
        and return the indices of the matched keypoints"""
        matched_indices = []
        non_matched_indices = []
        for idx, keypoint in enumerate(landmarks):
            min_computed_distance = np.inf
            min_id = -1
            for idx2, existing_keypoint in enumerate(landmarks_map):
                distance = np.sqrt((keypoint[0] - existing_keypoint[0])**2 + (keypoint[1] - existing_keypoint[1])**2)
                if distance < min_computed_distance and distance < threshold_match:
                    min_computed_distance = distance
                    min_id = idx2
                if distance < min_computed_distance:
                    min_computed_distance = distance

            if min_id != -1:
                matched_indices.append([idx, min_id, min_computed_distance])
            elif min_computed_distance > threshold_new:
                non_matched_indices.append(idx)

        return matched_indices, non_matched_indices

def process_landmarks(file_name, map_reflectors, distance_threshold=0.3, nn_threshold=4):
    # Read reflectors from the file
    my_reflectors = read_reflectors_from_file(file_name)
    
    # Match landmarks
    matched_indices, _ = match_landmarks_NN(None, my_reflectors, map_reflectors, distance_threshold, nn_threshold)
    
    # Calculate errors
    errors = [idx[2] for idx in matched_indices]
    for idx in matched_indices:
        print(f"Error{idx[0]}-{idx[1]}: {idx[2]:.5f}")
    
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    min_error = np.min(errors)
    return errors, mean_error, max_error, min_error

file_name = os.path.join(path, "example_scans", "landmarks_test.txt")

# Process the file
errors, mean_error, max_error, min_error = process_landmarks(file_name, map_reflectors)
delta_error = max_error - min_error

# Prepare the figure
plt.figure(figsize=(8, 6))

# Plot the error data
plt.scatter(np.arange(len(errors)), errors, c='r', label='Errors')
plt.axhline(y=mean_error, color='b', linestyle='--', label=f'Mean Error = {mean_error:.5f}')
plt.axhline(y=max_error, color='g', linestyle='--', label=f'Max Error = {max_error:.5f}')
plt.axhline(y=min_error, color='y', linestyle='--', label=f'Min Error = {min_error:.5f}')
#plt.vlines(x=-0.5, ymin=min_error, ymax=max_error, color='r', linestyle='-', label=f'Delta Error = {delta_error:.5f}')

# Set axis limits and labels
plt.ylim(0, max_error+ 0.005)
plt.grid()
plt.legend()
plt.title(f"DISTANCE ERROR OF MAPPED POSITIONS vs REAL POSITION")
plt.xlabel("Reflector Index")
plt.ylabel("Error")

# Show the figure
#plt.tight_layout()
plt.show()

