import numpy as np

class ProbabilisticOccupancyGrid:
    def __init__(self, resolution= 0.2, prior=0.5):
        self.resolution = resolution
        
        self.min_x = -1
        self.min_y = 0
        self.max_x = 1
        self.max_y = 3
        self.set_map_size(self.min_x, self.min_y, self.max_x, self.max_y)
        self.gridMap = np.full((self.width, self.height), prior)

    def world2grid(self, x, y):
        x = int((x - self.min_x) / self.resolution)
        y = int((y - self.min_y) / self.resolution)
        return x, y
    
    def grid2world(self, x, y):
        x = x * self.resolution + self.min_x
        y = y * self.resolution + self.min_y
        return x, y
    
    def set_map_size(self, min_x, min_y, max_x, max_y):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.width = int((max_x - min_x) / self.resolution)
        self.height = int((max_y - min_y) / self.resolution)
        self.map = np.zeros((self.width, self.height))

    def relative_to_global_point_cloud(self, robot_pose, relative_point_cloud):
        """
        Transform relative point cloud to global point cloud using robot pose
        Input:
            robot_pose: [x, y, theta]
            relative_point_cloud: [[x1, y1], [x2, y2], ...]
        """
        x, y, theta = robot_pose
        H = self.homogeneous_transformation(robot_pose)
        
    
    def relative2global_pointCloud(self, pose, points):
        x, y, theta = pose
        H =np.array([[np.cos(theta), -np.sin(theta), x], [np.sin(theta), np.cos(theta), y], [0, 0, 1]])
        points = np.array(points)
        points = np.vstack((points.T, np.ones(points.shape[0])))
        return np.dot(H, points)

    def update_map(self, robot_pose, relative_point_cloud):
        """
        Update map using robot pose and relative point cloud
        Input:
            robot_pose: [x, y, theta]
            relative_point_cloud: [[x1, y1], [x2, y2], ...]
        """
        global_point_cloud = self.relative2global_pointCloud(robot_pose, relative_point_cloud)
        # Now we have the global point cloud, we can update the map using bayesian update and ray casting
        ##TODO: Implement this function



Map = ProbabilisticOccupancyGrid()

print(Map.gridMap)
print(Map.world2grid(0, 0))