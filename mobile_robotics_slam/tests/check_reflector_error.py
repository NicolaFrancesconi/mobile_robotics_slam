import numpy as np
import matplotlib.pyplot as plt



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

map_reflectors = read_reflectors_from_file("reflector_map_2.txt")


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

# List of files to process
files = [
    #"landmarks_lag_1.txt",
    # "landmarks_lag_2.txt",
    # "landmarks_lag_5.txt",
    # "landmarks_lag_10.txt",
    "landmarks_test.txt"

]

# Prepare the figure for subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 grid for 4 files
axes = axes.flatten()  # Flatten to iterate over the axes array

# Process each file and plot on a subplot
for i, file_name in enumerate(files):
    errors, mean_error, max_error, min_error = process_landmarks(file_name, map_reflectors)
    delta_error = max_error - min_error
    
    # Plot on the corresponding subplot
    ax = axes[i]
    ax.scatter(np.arange(len(errors)), errors, c='r')
    ax.axhline(y=mean_error, color='b', linestyle='--', label=f'Mean Error = {mean_error:.5f}')
    ax.vlines(x=-0.5, ymin= min_error, ymax=max_error, color='r', linestyle='-', label=f'Delta Error = {delta_error:.5f}')
    ax.axhline(y=max_error, color='g', linestyle='--', label=f'Max Error = {max_error:.5f}')
    ax.axhline(y=min_error, color='y', linestyle='--', label=f'Min Error = {min_error:.5f}')

    ax.set_ylim(0, max_error + 0.1)
    ax.grid()
    ax.legend()
    ax.set_title(f"Error Plot for {file_name}")
    ax.set_xlabel("Index")
    ax.set_ylabel("Error")

# Adjust layout and show the figure
plt.tight_layout()
plt.show()

