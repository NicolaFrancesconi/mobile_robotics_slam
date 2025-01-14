import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
import open3d as o3d
from sklearn.metrics import euclidean_distances
from scipy.optimize import linear_sum_assignment

# Necessary to run the script from visual studio code
path = __file__
file_location_subfolders = 3 #Number of folder to go up to reach root of package
for _ in range(file_location_subfolders):
    path = os.path.dirname(path)
sys.path.insert(0, path)

from mobile_robotics_slam.Extractors.Corners.CornerExtractor import CornerExtractor
corner_extractor = CornerExtractor()

# Set the parameters of the Corner Extractor
min_corner_angle = 70
max_corner_angle = 110
max_intersecton_distance = 0.8
corner_extractor.set_corner_params(max_intersecton_distance, min_corner_angle, max_corner_angle)

# Set the parameters of the Adaptive Segment Detector
sigma_ranges = 0.15
lambda_angle = 10
merge_distance = 0.07
min_points_density = 2
min_segment_length = 0.3
corner_extractor.set_detector_params(sigma_ranges, lambda_angle, merge_distance, min_points_density, min_segment_length)

# Set the parameters of the Segment Handler
epsilon = 0.1
min_density_after_segmentation = 8
min_length_after_segmentation = 0.12
corner_extractor.set_handler_params(epsilon, min_density_after_segmentation, min_length_after_segmentation)


# Prepare the data from the reference scan
ranges = np.loadtxt("scan1.txt") # Load the ranges from the reference scan
field_of_view = 2 * np.pi # Field of view of the laser scan
angle_min = -np.pi # Minimum angle of the laser scan
angles = [angle_min + i * field_of_view / len(ranges) for i in range(len(ranges))]

start = time.time()

corner_extractor.extract_corners(ranges, field_of_view, angle_min)
end = time.time()
corner_extractor.plot_corners()
extracted_corners1 = corner_extractor.get_corners()

ranges = np.loadtxt("scan2.txt") # Load the ranges from the reference scan

corner_extractor.extract_corners(ranges, field_of_view, angle_min)
print("Time taken to extract corners: ", end-start)
corner_extractor.plot_corners()
extracted_corners2 = corner_extractor.get_corners()

extracted = []
extracted_angles = []
i = 0
print("Extracted Corners from Reference Scan")
for corner in extracted_corners1:
    extracted.append(corner.get_position())
    print(f"Corner {i}:",np.rad2deg(corner.angle), corner.get_position())
    extracted_angles.append(np.rad2deg(corner.angle))
    i += 1
extracted = np.array(extracted)

mapped = []
mapped_angles = []
print("Extracted Corners from MAP")
i = 0
for corner in extracted_corners2:
    mapped.append(corner.get_position())
    print(f"Corner {i}:",np.rad2deg(corner.angle), corner.get_position())
    mapped_angles.append(np.rad2deg(corner.angle))
    i += 1
mapped = np.array(mapped)


#randomly remove some of the extracted corners and relative angles
# np.random.shuffle(extracted)
# np.random.shuffle(extracted_angles)
extracted = extracted[:int(len(extracted)*0.5)]
extracted_angles = extracted_angles[:int(len(extracted_angles)*0.5)]
# Add some random point and random angle to extracted
extracted = np.vstack((extracted, np.array([[0.5, -0.5], [1, 0.5]])))
extracted_angles = np.hstack((extracted_angles, np.array([0, 0])))

mapped = mapped[:int(len(mapped)*8)]
mapped_angles = mapped_angles[:int(len(mapped_angles)*8)]





plt.figure()
plt.title("Extracted Corners from Reference Scan")
plt.scatter(extracted[:, 0], extracted[:, 1], c='r', label='Reference Scan')
plt.scatter(mapped[:, 0], mapped[:, 1], c='b', label='Current Scan')
plt.legend()
plt.show()

import networkx as nx
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt


def construct_compatibility_graph(extracted, mapped, extracted_angles, mapped_angles, distance_tolerance, angle_tolerance):
    """
    Constructs a compatibility graph where:
      - Nodes represent potential matches between extracted and mapped points.
      - Edges represent consistency between matches.
    """
    G = nx.Graph()
    matches = []

    # Create nodes: potential matches between extracted and mapped
    for i, p1 in enumerate(extracted):
        for j, p2 in enumerate(mapped):
            # Check if angles are compatible
            if abs(extracted_angles[i] - mapped_angles[j]) <= angle_tolerance:
                G.add_node((i, j), extracted_index=i, mapped_index=j)
                matches.append((i, j))

    # Create edges: consistency between matches
    for (i1, j1), (i2, j2) in combinations(matches, 2):
        # Ensure matches involve different points
        if i1 != i2 and j1 != j2:
            # Check geometric consistency
            d1_extracted = np.linalg.norm(extracted[i1] - extracted[i2])
            d1_mapped = np.linalg.norm(mapped[j1] - mapped[j2])
            if abs(d1_extracted - d1_mapped) <= distance_tolerance:
                G.add_edge((i1, j1), (i2, j2))

    return G


def find_maximum_clique(compatibility_graph):
    """
    Finds the maximum clique in the compatibility graph.
    """
    cliques = list(nx.find_cliques(compatibility_graph))  # Find all cliques
    for clique in cliques:
        print("Clique:", clique)
    max_clique = max(cliques, key=len) if cliques else []  # Select the largest
    return max_clique


tempo = time.time()
distance_tolerance = 0.05  # Maximum allowed distance difference
angle_tolerance = 3  # Maximum allowed angle difference

# Construct compatibility graph
compatibility_graph = construct_compatibility_graph(extracted, mapped, extracted_angles, mapped_angles, distance_tolerance, angle_tolerance)

# Find maximum clique
max_clique = find_maximum_clique(compatibility_graph)
print("Maximum Clique:", max_clique)

print("Time taken to find maximum clique: ", time.time()-tempo)
# Visualize matches
plt.figure()
plt.title("Matched Corners with Maximum Clique")
plt.scatter(extracted[:, 0], extracted[:, 1], c='r', label='Extracted Corners')
plt.scatter(mapped[:, 0], mapped[:, 1], c='b', label='Mapped Corners')

for i, j in max_clique:
    plt.plot([extracted[i, 0], mapped[j, 0]], [extracted[i, 1], mapped[j, 1]], 'g--')

plt.legend()
plt.show()


def match_distances(local_map_pairwise_distance, partial_global_map_pairwise_distance):    #TODO: change partial_global_map_pairwise_distance to global_map_pairwise_distance
        '''Match the distances between the reflectors and the ones in the map'''
        matches_l_g = [] #match pairs local-global
        d_match = 0
        matched_rows = 0
        match = False

        f = []

        #secret code
        for row_id_v, v_row in enumerate(local_map_pairwise_distance[:]):
            for row_id_m, m_row in enumerate(partial_global_map_pairwise_distance[:]):
                d_match = -1    # -1 because the diagonal is not considered being 0
                for dv in v_row:
                    for dm in m_row:
                        diff = abs(dv - dm)
                        if diff <= 0.6:
                            d_match += 1
                if d_match >= 1 * (len(local_map_pairwise_distance) - 1):
                    matched_rows += 1
                    matches_l_g = matches_l_g + [[row_id_v,row_id_m]]

        matches_l_g = np.array(matches_l_g)

        if matched_rows >= 2:
            unique_l, counts_l = np.unique(matches_l_g.T[0], return_counts=True)
            to_remove_l = unique_l[counts_l > 1], counts_l[counts_l > 1]
            n_to_remove_l = np.sum(to_remove_l[1])
            unique_g, counts_g = np.unique(matches_l_g.T[1], return_counts=True)
            to_remove_g = unique_g[counts_g > 1], counts_g[counts_g > 1]
            
            #check presence of non univocal matches
            if len(to_remove_l) != 0:    
                for el in matches_l_g:
                    if len(matches_l_g) - n_to_remove_l > 1:
                        if not el[0] in to_remove_l[0]:
                            f = f + [el]
                    else :
                        if not el[0] in to_remove_l[0] or not el[1] in to_remove_g[0]:
                            f = f + [el]
                matches_l_g = np.asarray(f)
                match = True
            elif len(matches_l_g) > 1:
                match = True   
            else:
                match = False         

        return match, matches_l_g


mpped_distances = euclidean_distances(mapped)
extracted = mapped + np.array([0.5, -0.5])

extracted = extracted[:int(len(extracted)*0.8)]

extracted_distances = euclidean_distances(extracted)

print("Extracted Distances:")
print(extracted_distances)
print("Mapped Distances:")
print(mpped_distances)

if len(mapped) > len(extracted):
    match, matches = match_distances(extracted_distances, mpped_distances)
else:
    match, matches = match_distances(mpped_distances, extracted_distances)

print("Match:", matches)
