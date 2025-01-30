import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import networkx as nx
from itertools import combinations
import time


path = __file__
file_location_subfolders = 4 #Number of folder to go up to reach root of package
for _ in range(file_location_subfolders):
    path = os.path.dirname(path)
sys.path.insert(0, path)

from mobile_robotics_slam.Optimizer.GTSAMGraphOptimizer import GTSAMGraphOptimizer
import mobile_robotics_slam.Params.simulation_params as params

class PoseVertex:
    def __init__(self, id, x, y, theta, point_cloud):
        self.id = id
        self.x = x
        self.y = y
        self.theta = theta
        self.point_cloud = point_cloud

class LandmarkVertex:
    def __init__(self, id, object):
        # Check if Object has attribute x and y and get_position method
        if not hasattr(object, 'x') or not hasattr(object, 'y'):
            raise ValueError(f"Object: {type(object).__name__} must have both x and y attributes")
        if not hasattr(object, "get_position"):
            raise ValueError(f"Object: {type(object).__name__} must have get_position method")
        else: print("Added Landmark of type: ", type(object).__name__ , " with id: ", id)
        self.id = id
        self.object = object 

class GraphHandler:
    def __init__(self):
        print("GTSAM Graph Handler Initialized")
        self.graph_optimizer = GTSAMGraphOptimizer()
        self.pose_vertices = []
        self.landmark_vertices = []
        self.pose_base_id = 10000 # Base ID for Pose Vertices to avoid conflicts with Landmark IDs

    def get_mapped_landmarks(self):
        mapped_reflectors = []
        mapped_corners = []
        for landmark in self.landmark_vertices:
            if type(landmark.object).__name__ == "Corner":
                position = landmark.object.get_position()
                angle = landmark.object.angle
                id = landmark.id
                mapped_corners.append([position[0], position[1], angle, id])
            if type(landmark.object).__name__ == "Reflector":
                position = landmark.object.get_position()
                id = landmark.id
                mapped_reflectors.append([position[0], position[1], id])
        return mapped_corners, mapped_reflectors
    
    def categorize_extracted_landmarks(self, landmarks):
        extracted_corners = []
        extracted_reflectors = []
        for idx ,landmark in enumerate(landmarks):
            if type(landmark).__name__ == "Corner":
                position = landmark.get_position()
                angle = landmark.angle
                extracted_corners.append([position[0], position[1], angle, idx])
            if type(landmark).__name__ == "Reflector":
                position = landmark.get_position()
                extracted_reflectors.append([position[0], position[1], idx])
        return extracted_corners, extracted_reflectors
    

    def match_reflectors(self, extracted_reflectors, mapped_reflectors):
        if len(mapped_reflectors) == 0:
                non_matched_idxs = [i for i in range(len(extracted_reflectors))]
                return False, [], non_matched_idxs
        if len(extracted_reflectors) == 0:
                return False, [], []
        match = False
        distance_tolerance = params.REFLECTOR_RELATIVE_DISTANCE_TOLERANCE
        neighbor_distance = params.REFLECTOR_NEIGHBOR_MAX_DISTANCE
        compatibility_graph = construct_reflector_compatibility_graph(extracted_reflectors, mapped_reflectors, distance_tolerance, neighbor_distance)
        matched_idxs, unique = find_maximum_clique(compatibility_graph)
        non_matched_idxs = [i for i in range(len(extracted_reflectors)) if i not in [idx[0] for idx in matched_idxs]]
        if unique:
            if len(matched_idxs) > 0:
                    match = True
        elif len(matched_idxs) > 2:
            match = True
        return match, matched_idxs, non_matched_idxs
    
    def match_corners(self, extracted_corners, mapped_corners):
        if len(mapped_corners) == 0:
                non_matched_idxs = [i for i in range(len(extracted_corners))]
                return False, [], non_matched_idxs
        if len(extracted_corners) == 0:
                return False, [], []
        match = False
        relative_distance_tolerance = params.CORNER_RELATIVE_DISTANCE_TOLERANCE
        relative_angle_tolerance = params.CORNER_RELATIVE_ANGLE_TOLERANCE
        neighbor_distance = params.CORNER_NEIGHBOR_MAX_DISTANCE
        compatibility_graph = construct_corner_compatibility_graph(extracted_corners, mapped_corners, relative_distance_tolerance, relative_angle_tolerance, neighbor_distance)
        matched_idxs, unique = find_maximum_clique(compatibility_graph)
        non_matched_idxs = [i for i in range(len(extracted_corners)) if i not in [idx[0] for idx in matched_idxs]]
        if unique:
            if len(matched_idxs) > 0:
                    match = True
        elif len(matched_idxs) > 2:
            match = True
        return match, matched_idxs, non_matched_idxs
    
    def add_matched_reflectors_edge(self, matched_indices, pose, landmarks, pose_id, mapped_reflectors, extracted_reflectors):
        for idx_E, idx_M in matched_indices:
            id_E = extracted_reflectors[idx_E][2]
            id_M = mapped_reflectors[idx_M][2]
            position = landmarks[id_E].get_position()
            self.graph_optimizer.add_pose_landmark_edge_2D(pose_id, id_M, position)

    def add_matched_corners_edge(self, matched_indices, pose, landmarks, pose_id, mapped_corners, extracted_corners):
        for idx_E, idx_M in matched_indices:
            id_E = extracted_corners[idx_E][3]
            id_M = mapped_corners[idx_M][3]
            position = landmarks[id_E].get_position()
            self.graph_optimizer.add_pose_landmark_edge_2D(pose_id, id_M, position)

    def add_non_matched_corners(self, extracted_corners, pose_id, landmarks, non_matched_indices, mapped_corners):
        for idx in non_matched_indices:
            id = len(self.landmark_vertices)
            position = landmarks[extracted_corners[idx][3]].get_position()
                #Check if the landmark if no other landmark is close to it
            mapped_corners = np.array(mapped_corners)
            if len(mapped_corners) > 0:
                distances = np.linalg.norm(mapped_corners[:, :2] - position, axis=1)
            else:
                distances = np.array([1000])
            if np.min(distances) > params.CORNER_ADD_DISTANCE_THRESHOLD:
                self.graph_optimizer.add_pose_landmark_edge_2D(pose_id, id, position)
                self.landmark_vertices.append(LandmarkVertex(id, landmarks[extracted_corners[idx][3]]))

    def add_non_matched_reflectors(self, extracted_reflectors, pose_id, landmarks, non_matched_indices, mapped_reflectors):
        for idx in non_matched_indices:
            id = len(self.landmark_vertices)
            position = landmarks[extracted_reflectors[idx][2]].get_position()
            #Check if the landmark if no other landmark is close to it
            mapped_reflectors = np.array(mapped_reflectors)
            if len(mapped_reflectors) > 0:
                distances = np.linalg.norm(mapped_reflectors[:, :2] - position, axis=1)
            else:
                distances = np.array([1000])
            if np.min(distances) > params.REFLECTOR_ADD_DISTANCE_THRESHOLD:
                self.graph_optimizer.add_pose_landmark_edge_2D(pose_id, id, position)
                self.landmark_vertices.append(LandmarkVertex(id, landmarks[extracted_reflectors[idx][2]]))

        

    def add_to_graph(self, pose, pointcloud, landmarks):
        pose_id = self.pose_base_id + len(self.pose_vertices)
        if len(self.pose_vertices) == 0:
            self.graph_optimizer.add_prior_pose_2D(pose_id, pose)
            self.pose_vertices.append(PoseVertex(pose_id, pose[0], pose[1], pose[2], pointcloud))
            for id,landmark in enumerate(landmarks):
                self.graph_optimizer.add_pose_landmark_edge_2D(pose_id, id, landmark.get_position())
                self.landmark_vertices.append(LandmarkVertex(id, landmark))
            return pose
        
        robot_pose = np.array([pose[0], pose[1], pose[2]])
        mapped_corners, mapped_reflectors = self.get_mapped_landmarks()
        extracted_corners , extracted_reflectors = self.categorize_extracted_landmarks(landmarks)
        match_ref, matched_indices_ref, non_matched_indices_ref = self.match_reflectors(extracted_reflectors, mapped_reflectors)
        match_cor, matched_indices_cor, non_matched_indices_cor = self.match_corners(extracted_corners, mapped_corners)
        
        self.add_pose_vertex(pose, pointcloud, pose_id)
        
        if match_ref:
            self.add_matched_reflectors_edge(matched_indices_ref, robot_pose, landmarks, pose_id, mapped_reflectors, extracted_reflectors)

        if match_cor:
            self.add_matched_corners_edge(matched_indices_cor, robot_pose, landmarks, pose_id, mapped_corners, extracted_corners)

        # Add Landmark Vertices Of Non Matched Landmarks
        self.add_non_matched_corners(extracted_corners, pose_id, landmarks, non_matched_indices_cor, mapped_corners)
        self.add_non_matched_reflectors(extracted_reflectors, pose_id, landmarks, non_matched_indices_ref, mapped_reflectors)

        if  match_cor or match_ref :
            self.optimize_graph()
            
        last_pose = self.graph_optimizer.get_pose_2D(pose_id)
        last_pose = np.array([last_pose[0], last_pose[1], last_pose[2]])

        return last_pose
               
    def add_pose_vertex(self, pose, pointcloud, pose_id):
        id = pose_id
        self.pose_vertices.append(PoseVertex(id, pose[0], pose[1], pose[2], pointcloud))
        self.graph_optimizer.add_odometry_edge_2D(id-1, id, pose)
            

    def optimize_graph(self):
        self.graph_optimizer.optimize(10)

        # Update Pose Vertices with Optimized Values knowing the id of the optimized vertices
        for pose in self.pose_vertices:
            optimized_pose = self.graph_optimizer.get_pose_2D(pose.id)
            pose.x = optimized_pose[0]
            pose.y = optimized_pose[1]
            pose.theta = optimized_pose[2]
        
        # Update Landmark Vertices with Optimized Values knowing the id of the optimized vertices
        for landmark in self.landmark_vertices:
            optimized_landmark = self.graph_optimizer.get_landmark_2D(landmark.id)
            landmark.object.x = optimized_landmark[0]
            landmark.object.y = optimized_landmark[1]

    def get_optimized_poses_and_landmarks(self):
        poses = []
        point_clouds = []
        landmarks = []
        for pose in self.pose_vertices:
            poses.append(np.array([pose.x, pose.y, pose.theta]))
            point_clouds.append(pose.point_cloud)
        for landmark in self.landmark_vertices:
            landmarks.append(landmark.object.get_position())
        return poses, point_clouds, landmarks

    
    
def construct_corner_compatibility_graph(extracted, mapped, distance_tolerance, angle_tolerance, neighbor_distance):
    """
    Optimized construction of the compatibility graph.
    """
    G = nx.Graph()
    matches = []
    extracted = np.array(extracted) # [x, y, angle]
    mapped = np.array(mapped) # [x, y, angle]
    mapped_angles = mapped[:, 2]
    extracted_angles = extracted[:, 2]
    
    angle_diff_matrix = np.abs(extracted_angles[:, None] - mapped_angles[None, :])
    distance_matrix = np.linalg.norm(extracted[:, :2][:, None] - mapped[:, :2][None, :], axis=2)
    # Find valid matches based on angle tolerance
    valid_matches = np.where((angle_diff_matrix <= angle_tolerance) & (distance_matrix <= neighbor_distance))

    # Add nodes for each valid match
    for i, j in zip(valid_matches[0], valid_matches[1]):
        G.add_node((i, j), extracted_index=i, mapped_index=j, distance=distance_matrix[i, j])
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
        G.add_node((i, j), extracted_index=i, mapped_index=j, distance=distance_matrix[i, j])
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

    
