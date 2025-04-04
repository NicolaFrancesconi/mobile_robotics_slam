import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
import matplotlib.patches as mpatches

path = __file__
file_location_subfolders = 3 #Number of folder to go up to reach root of package
for _ in range(file_location_subfolders):
    path = os.path.dirname(path)
sys.path.insert(0, path)

from mobile_robotics_slam.Extractors.Reflectors.ReflectorExtractor import ReflectorExtractor

def construct_reflector_compatibility_graph(extracted, mapped, distance_tolerance, neighbor_distance):
    """
    Optimized construction of the compatibility graph.
    """
    G = nx.Graph()
    matches = []
    
    # Extract positions and angles for KDTree
    extracted = np.array(extracted) # [x, y]
    mapped = np.array(mapped) # [x, y]
    distance_matrix = np.linalg.norm(extracted[:, :2][:, None] - mapped[:, :2][None, :], axis=2)
    # Find valid matches based global distance tolerance
    valid_matches = np.where((distance_matrix <= neighbor_distance))
    # Add nodes for each valid match
    for i, j in zip(valid_matches[0], valid_matches[1]):
        G.add_node((i, j), extracted_index=i, mapped_index=j, distance=distance_matrix[i, j], pos=mapped[j])
        matches.append((i, j))

    # Create edges based on consistency between matches
    for idx1, idx2 in combinations(range(len(matches)), 2):
        i1, j1 = matches[idx1]
        i2, j2 = matches[idx2]

        # Ensure matches involve different points
        if i1 != i2 and j1 != j2:
            # Check geometric consistency between extracted and mapped matches
            extracted_diff = np.linalg.norm(np.array(extracted[i1][:2]) - np.array(extracted[i2][:2]))
            mapped_diff = np.linalg.norm(np.array(mapped[j1][:2]) - np.array(mapped[j2][:2]))
            if abs(extracted_diff - mapped_diff) <= distance_tolerance:  # Adjust tolerance as needed
                G.add_edge((i1, j1), (i2, j2))

    return G
    
def find_maximum_clique(compatibility_graph):
    """
    Finds the maximum clique in the compatibility graph.
    """
    unique = True
    cliques = list(nx.find_cliques(compatibility_graph))  # Find all cliques
    max_length = max(len(c) for c in cliques) if cliques else 0
    max_clique = max(cliques, key=len) if cliques else []  # Select the largest
    max_cliques = [c for c in cliques if len(c) == max_length]
    # If there are multiple cliques with the same size, select the one with the smallest distance
    print("Cliques", max_cliques)
    if len(max_cliques) > 1:
        print("Multiple Cliques with same size")
        max_clique = min(max_cliques, key=lambda c: sum(compatibility_graph.nodes[n]['distance'] for n in c))
        unique = False
    return max_clique, unique

import networkx as nx
import matplotlib.pyplot as plt

def visualize_maximum_clique(compatibility_graph, max_clique):
    """
    Visualizes the correspondence graph and highlights the maximum clique.
    """
    pos = nx.circular_layout(compatibility_graph)  # Compute positions for visualization
    #pos = nx.kamada_kawai_layout(compatibility_graph)
    plt.figure()
    
    # Draw the full graph
    nx.draw(compatibility_graph, pos, with_labels=True, node_color='Cyan', edge_color='gray', node_size=500)

    # Highlight the maximum clique
    nx.draw_networkx_nodes(compatibility_graph, pos, nodelist=max_clique, node_color='orange', node_size=700)
    nx.draw_networkx_edges(compatibility_graph, pos, edgelist=[(u, v) for u in max_clique for v in max_clique if u != v], edge_color='orange', width=2)
    legend_patches = [
        mpatches.Patch(color="lightgray", label="Compatibility Edges"),
        mpatches.Patch(color="Orange", label="Maximum Clique"),
        mpatches.Patch(color="Cyan", label="Correspondence Nodes"),
    ]
    plt.title("Correspondence Graph with Maximum Clique Highlighted")
    plt.legend(handles=legend_patches)
    plt.show()

# Example usage:
# max_clique, _ = find_maximum_clique(compatibility_graph)
# visualize_maximum_clique(compatibility_graph, max_clique)



#Test Using Data from intensity_scan.txt

# Load Data
data1 = np.loadtxt(os.path.join(path, "example_scans", "CliqueScan1.txt"))
scan_ranges1 = data1[:, 0]
scan_intensities1 = data1[:, 1]
angle_min = -np.pi
field_of_view = 2*np.pi

data2 = np.loadtxt(os.path.join(path, "example_scans", "CliqueScan2.txt"))
scan_ranges2 = data2[:, 0]
scan_intensities2 = data2[:, 1]


# Extract Reflectors
reflector_extractor = ReflectorExtractor()
reflector_extractor.max_range_extraction = 30

robot_pose = np.zeros(3)
reflector_extractor.extract_reflectors(scan_ranges1, scan_intensities1, field_of_view, angle_min, robot_pose)
landmarks = reflector_extractor.get_reflectors()
mapped_reflector = np.array([landmark.get_position() for landmark in landmarks])

reflector_extractor.extract_reflectors(scan_ranges2, scan_intensities2, field_of_view, angle_min, robot_pose)
landmarks = reflector_extractor.get_reflectors()
extracted_reflector = np.array([landmark.get_position() for landmark in landmarks])


print("Mapped Reflector: ", mapped_reflector)
print("Extracted Reflector: ", extracted_reflector)

#Plot the reflectors in the scan

angles = np.linspace(angle_min, angle_min + field_of_view, len(scan_ranges1))

points1 = np.array([scan_ranges1 * np.cos(angles), scan_ranges1 * np.sin(angles)])
points2 = np.array([scan_ranges2 * np.cos(angles), scan_ranges2 * np.sin(angles)])

plt.figure(figsize=(12, 8))
plt.scatter(points1[0], points1[1], s=2, label="Map", color="red")   
plt.scatter(points2[0], points2[1], s=2, label="Current Scan", color="blue")
plt.scatter(mapped_reflector[:,0], mapped_reflector[:,1], s=40, label="Mapped Reflector", color="magenta", marker="o", edgecolors="black")
for i, (x, y) in enumerate(mapped_reflector):
    plt.text(x, y, f"{i}", fontsize=10, ha="right", va="bottom", color="black")
plt.scatter(extracted_reflector[:,0], extracted_reflector[:,1], s=40, label="Extracted Reflector", color="yellow", marker="o", edgecolors="black")
for i, (x, y) in enumerate(extracted_reflector):
    plt.text(x, y, f"{i}", fontsize=10, ha="right", va="bottom", color="black")
plt.axis('equal')
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("Extracted and Mapped Reflectors")
plt.legend()
plt.show()

# Construct the compatibility graph
distance_tolerance = 0.05
neighbor_distance = 15
compatibility_graph = construct_reflector_compatibility_graph(extracted_reflector, mapped_reflector, distance_tolerance, neighbor_distance)

# Find the maximum clique
max_clique, unique = find_maximum_clique(compatibility_graph)
print("Maximum Clique: ", max_clique)

# Plot the maximum clique in the compatibility graph and the reflectors in the scan 
plt.figure(figsize=(12, 8))
plt.scatter(points1[0], points1[1], s=2, label="Map", color="red")
plt.scatter(points2[0], points2[1], s=2, label="Current Scan", color="blue")
plt.scatter(mapped_reflector[:,0], mapped_reflector[:,1], s=40, label="Mapped Reflector", color="magenta", marker="o", edgecolors="black")
for i, (x, y) in enumerate(mapped_reflector):
    plt.text(x, y, f"{i}", fontsize=10, ha="right", va="bottom", color="black")
plt.scatter(extracted_reflector[:,0], extracted_reflector[:,1], s=40, label="Extracted Reflector", color="yellow", marker="o", edgecolors="black")
for i, (x, y) in enumerate(extracted_reflector):
    plt.text(x, y, f"{i}", fontsize=10, ha="right", va="bottom", color="black")

a=0
for i, j in max_clique:
    
    plt.plot([extracted_reflector[i, 0], mapped_reflector[j, 0]], [extracted_reflector[i, 1], mapped_reflector[j, 1]], color="orange",  label="Matched Reflector" if a==0 else "")
    a+=1
plt.axis('equal')
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("Maximum Clique Matching")
plt.legend()
plt.show()

visualize_maximum_clique(compatibility_graph, max_clique)




robot_pose1 = np.array([0.0, 0.0, 0.4])

robot_pose2 = np.array([-5.0, -2., 0.0])

reflector_extractor.extract_reflectors(scan_ranges1, scan_intensities1, field_of_view, angle_min, robot_pose1)
landmarks = reflector_extractor.get_reflectors()
mapped_reflector = np.array([landmark.get_position() for landmark in landmarks])

reflector_extractor.extract_reflectors(scan_ranges2, scan_intensities2, field_of_view, angle_min, robot_pose2)
landmarks = reflector_extractor.get_reflectors()
extracted_reflector = np.array([landmark.get_position() for landmark in landmarks])


points1 = np.array([scan_ranges1 * np.cos(angles + robot_pose1[2]) + robot_pose1[0],
                     scan_ranges1 * np.sin(angles + robot_pose1[2])+ robot_pose1[1]])

points2 = np.array([scan_ranges2 * np.cos(angles + robot_pose2[2]) + robot_pose2[0],
                        scan_ranges2 * np.sin(angles + robot_pose2[2])+ robot_pose2[1]]) 

plt.figure(figsize=(12, 8))
plt.scatter(points1[0], points1[1], s=2, label="Map", color="red")
plt.scatter(points2[0], points2[1], s=2, label="Current Scan", color="blue")
plt.scatter(mapped_reflector[:,0], mapped_reflector[:,1], s=120, label="Mapped Reflector", color="magenta", marker="o", edgecolors="black")
plt.scatter(extracted_reflector[:,0], extracted_reflector[:,1], s=40, label="Extracted Reflector", color="yellow", marker="o", edgecolors="black")
# for i, (x, y) in enumerate(extracted_reflector):
#     plt.text(x, y, f"{i}", fontsize=10, ha="right", va="bottom", color="black")
# for i, (x, y) in enumerate(mapped_reflector):
#     plt.text(x, y, f"{i}", fontsize=10, ha="left", va="bottom", color="black")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.axis('equal')
plt.title("Aligned Extracted and Mapped Reflectors")
plt.legend()
plt.show()
