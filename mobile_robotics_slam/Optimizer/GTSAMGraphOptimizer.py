
## Not Tested yet  ##


import gtsam
from gtsam import Pose2, Point2, noiseModel
import numpy as np


class GTSAMGraphOptimizer:
    def __init__(self):
        # Create a factor graph
        self.graph = gtsam.NonlinearFactorGraph()
        # Initial estimate for all vertices
        self.initial_estimate = gtsam.Values()
        # Optimization results (set after optimization)
        self.result = None

    def optimize(self, max_iterations=20):
        """
        Optimize the graph using Levenberg-Marquardt
        """
        print(f"Optimizing Graph with {self.graph.size()} factors")
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate)
        self.result = optimizer.optimizeSafely()
        print(f"Optimization complete after {max_iterations} iterations.")

    def add_pose_vertex_2D(self, id, pose, fixed=False):
        """
        Add a pose vertex to the graph with the given id and pose
        """
        if fixed:
            # Add a prior factor to fix the pose
            prior_noise = noiseModel.Diagonal.Sigmas(np.array([1e-5, 1e-5, 1e-5]))
            self.graph.add(gtsam.PriorFactorPose2(id, Pose2(pose[0], pose[1], pose[2]), prior_noise))

        # Add the initial estimate for the pose
        self.initial_estimate.insert(id, Pose2(pose[0], pose[1], pose[2]))
        print(f"Added pose vertex with id {id} and pose {pose}")

    def add_landmark_vertex_2D(self, id, position, fixed=False):
        """
        Add a landmark vertex to the graph with the given id and position
        """
        if fixed:
            # Add a prior factor to fix the landmark position
            prior_noise = noiseModel.Diagonal.Sigmas(np.array([1e-5, 1e-5]))
            self.graph.add(gtsam.PriorFactorPoint2(id, Point2(position[0], position[1]), prior_noise))

        # Add the initial estimate for the landmark
        self.initial_estimate.insert(id, Point2(position[0], position[1]))
        print(f"Added landmark vertex with id {id} and position {position}")

    def add_odometry_edge_2D(self, vertex_id1, vertex_id2, relative_transform, information_matrix):
        """
        Add an odometry edge between two pose vertices
        """
        # Create a noise model from the information matrix
        covariance = np.linalg.inv(information_matrix)
        noise = noiseModel.Gaussian.Covariance(covariance)
        self.graph.add(gtsam.BetweenFactorPose2(vertex_id1, vertex_id2, Pose2(*relative_transform), noise))
        print(f"Added odometry edge between {vertex_id1} and {vertex_id2}")

    def add_pose_landmark_edge_2D(self, pose_id, landmark_id, relative_measurement, information_matrix):
        """
        Add an edge between a pose and a landmark
        """
        # Create a noise model from the information matrix
        covariance = np.linalg.inv(information_matrix)
        noise = noiseModel.Gaussian.Covariance(covariance)
        bearing = gtsam.gtsam.Rot2.atan2(1, 0)
        print("Bearings: ", bearing)
        self.graph.add(gtsam.BearingRangeFactor2D(pose_id, landmark_id, bearing, np.linalg.norm(relative_measurement), noise))
        print(f"Added pose-landmark edge between {pose_id} and {landmark_id}")

    def get_pose_2D(self, id):
        """
        Get the pose of a vertex after optimization
        """
        if self.result is None:
            raise ValueError("Graph has not been optimized yet.")
        return self.result.atPose2(id).vector()

    def get_landmark_2D(self, id):
        """
        Get the position of a landmark after optimization
        """
        if self.result is None:
            raise ValueError("Graph has not been optimized yet.")
        return np.array(self.result.atPoint2(id))

    def get_all_poses_2D(self):
        """
        Get all optimized poses in the graph
        """
        if self.result is None:
            raise ValueError("Graph has not been optimized yet.")
        poses = []
        print("Results: ", self.graph)
        

    def get_all_landmarks_2D(self):
        """
        Get all optimized landmark positions in the graph
        """
        if self.result is None:
            raise ValueError("Graph has not been optimized yet.")
        landmarks = []
        for key in self.result.keys():
            pass
            # if isinstance(self.result.at(key), gtsam.Point2):
            #     landmarks.append((key, np.array(self.result.atPoint2(key))))
        return landmarks


optimizer = GTSAMGraphOptimizer()
optimizer.add_pose_vertex_2D(0, [0, 0, 0], fixed=True)
optimizer.add_pose_vertex_2D(1, [1, 0, 0])
optimizer.add_landmark_vertex_2D(2, [1, 1])
optimizer.add_odometry_edge_2D(0, 1, [1, 0, 0], np.eye(3))
optimizer.add_pose_landmark_edge_2D(1, 2, [0, 1], np.eye(2))
optimizer.optimize()
print("Optimized Poses:", optimizer.get_all_poses_2D())
print("Optimized Landmarks:", optimizer.get_all_landmarks_2D())