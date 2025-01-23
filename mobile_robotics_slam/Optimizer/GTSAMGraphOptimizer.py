import gtsam
from gtsam import Pose2, Point2, noiseModel
import numpy as np




class GTSAMGraphOptimizer:
    def __init__(self):
        # Create a factor graph
        self.graph = gtsam.NonlinearFactorGraph()
        # Initial estimate for all vertices
        self.initial_estimate = gtsam.Values()
        self.actual_estimate = gtsam.Values()

        self.landmarks_key = []
        self.poses_key = []

        self.PRIOR_POSE_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6, 1e-6, 1e-6]))
        self.PRIOR_LANDMARK_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6, 1e-6]))
        self.ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0, 1.0, 1.0]))
        self.MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01]))

    def set_PRIOR_POSE_NOISE(self, noise):
        gtsam_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([noise[0], noise[1], noise[2]]))
        self.PRIOR_POSE_NOISE = gtsam_noise

    def set_ODOMETRY_NOISE(self, noise):
        gtsam_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([noise[0], noise[1], noise[2]]))
        self.ODOMETRY_NOISE = gtsam_noise

    def set_MEASUREMENT_NOISE(self, noise):
        gtsam_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([noise[0], noise[1]]))
        self.MEASUREMENT_NOISE = gtsam_noise
    
    def set_PRIOR_LANDMARK_NOISE(self, noise):
        gtsam_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([noise[0], noise[1]]))
        self.PRIOR_LANDMARK_NOISE = gtsam_noise


    def add_prior_pose_2D(self,id: int, pose):
        """
        Add a prior pose factor to the graph
        Input:
            id: Pose Vertex ID
            pose: [x, y, theta] pose of the robot
        """
        self.graph.add(gtsam.PriorFactorPose2(id, Pose2(pose[0], pose[1], pose[2]), self.PRIOR_POSE_NOISE))
        # Add the initial estimate for the pose
        self.initial_estimate.insert(id, Pose2(pose[0], pose[1], pose[2]))
        self.actual_estimate.insert(id, Pose2(pose[0], pose[1], pose[2]))
        self.poses_key.append(id)
        #print(f"Added prior pose vertex with id {id} and pose {pose}")
        return id

    def add_prior_landmark_2D(self, id:int, position):
        """
        Add a prior factor to the graph
        Input:
            id: Landmark Vertex ID
            position: [x, y] position of the landmark
        """
        point = Point2(position[0], position[1])
        self.landmarks_key.append(id)
        self.graph.add(gtsam.PriorFactorPoint2(id, point, self.PRIOR_LANDMARK_NOISE))
        # Add the initial estimate for the landmark
        self.initial_estimate.insert(id, point)
        self.actual_estimate.insert(id, point)
        #print(f"Added prior landmark vertex with id {id} and position {position}")

    def add_odometry_edge_2D(self, id1, id2, robot_pose):
        """
        Add an odometry edge between two pose vertices
        Input:
            id1: Pose Vertex ID 1
            id2: Pose Vertex ID 2
            robot_pose: [x, y, theta] pose of the robot
        """
        if id1 not in self.poses_key:
            raise ValueError("Pose ID 1 not found in the graph")
        
        poseId1 = id1
        poseId2 = id2
        self.poses_key.append(poseId2)
        pose = Pose2(robot_pose[0], robot_pose[1], robot_pose[2]) 
        previous_pose = self.actual_estimate.atPose2(poseId1)
        relative_transform = previous_pose.between(pose)
        # Create a noise model from the information matrix
        self.graph.add(gtsam.BetweenFactorPose2(poseId1,poseId2, Pose2(relative_transform), self.ODOMETRY_NOISE))
        self.initial_estimate.insert(poseId2, pose)
        self.actual_estimate.insert(poseId2, pose)
        #print(f"Added odometry edge between {poseId1} and {poseId2}")


    def add_pose_landmark_edge_2D(self, pose_id, landmark_id, landmark_position):
        """
        Add an edge between a pose and a landmark
        """
        robot_pose = self.actual_estimate.atPose2(pose_id)
        point_pose = Point2(landmark_position[0], landmark_position[1])
        bearing = robot_pose.bearing(point_pose)
        range = robot_pose.range(point_pose)
        self.graph.add(gtsam.BearingRangeFactor2D(pose_id, landmark_id, bearing, range, self.MEASUREMENT_NOISE))
        
        # If the landmark is not in the initial estimate, add it
        if not self.initial_estimate.exists(landmark_id):
            self.initial_estimate.insert(landmark_id, point_pose)
            self.actual_estimate.insert(landmark_id, point_pose)
            self.landmarks_key.append(landmark_id)
            #print(f"Added landmark vertex with id {landmark_id} and position {landmark_position}")
        
        #print(f"Added pose-landmark edge between {pose_id} and {landmark_id}")

    
        

    def optimize(self, max_iterations=20):
        """
        Optimize the graph using Levenberg-Marquardt
        """
        print(f"Optimizing Graph with {self.graph.size()} factors")
        params = gtsam.LevenbergMarquardtParams()
        #params.setVerbosityLM("SUMMARY")
        params.setMaxIterations(max_iterations)
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.actual_estimate, params)
        self.actual_estimate = optimizer.optimize()
        return self.actual_estimate

    
    def get_pose_2D(self, id):
        """
        Get the pose of a Pose2 Factor
        """
        if id not in self.poses_key:
            raise ValueError("Pose ID not found in the graph")
        x = self.actual_estimate.atPose2(id).x()
        y = self.actual_estimate.atPose2(id).y()
        theta = self.actual_estimate.atPose2(id).theta()
        return np.array([x, y, theta])

    def get_landmark_2D(self, id):
        """
        Get the position of a landmark after optimization
        """
        if id not in self.landmarks_key:
            raise ValueError("Landmark ID not found in the graph")
        return self.actual_estimate.atPoint2(id)

    def get_all_poses_2D(self):
        """
        Get all optimized poses in the graph
        """
        poses = []
        for key in self.poses_key:
            poses.append([key, self.get_pose_2D(key)])
        return poses
        

    def get_all_landmarks_2D(self):
        """
        Get all optimized landmark positions in the graph
        """
        landmarks = []
        for key in self.landmarks_key:
            landmarks.append([key, self.get_landmark_2D(key)])
        return landmarks


