import os
import sys
import numpy as np
import matplotlib.pyplot as plt

path = __file__
file_location_subfolders = 3 #Number of folder to go up to reach root of package
for _ in range(file_location_subfolders):
    path = os.path.dirname(path)
sys.path.insert(0, path)

from mobile_robotics_slam.Optimizer.g2oGraphOptimizer import GraphOptimizer

def relative_transform(pose1, pose2):
    """
    Given two poses [x1, y1, theta1] and [x2, y2, theta2], 
    compute the relative transformation [dx, dy, dtheta].
    
    Args:
    - pose1: List or array [x1, y1, theta1] for the first pose.
    - pose2: List or array [x2, y2, theta2] for the second pose.
    
    Returns:
    - [dx, dy, dtheta]: The relative transformation from pose1 to pose2.
    """
    
    x1, y1, theta1 = pose1
    x2, y2, theta2 = pose2
    
    # Compute the relative angle (dtheta)
    dtheta = (theta2 - theta1) % (2 * np.pi)  # Normalize the angle to [0, 2*pi)
    
    # Compute the relative position (dx, dy) in the frame of pose1
    dx = (x2 - x1) * np.cos(theta1) + (y2 - y1) * np.sin(theta1)
    dy = -(x2 - x1) * np.sin(theta1) + (y2 - y1) * np.cos(theta1)
    
    return [dx, dy, dtheta]

# Create a graph optimizer
graph_optimizer = GraphOptimizer()

# Ground truth poses of a circular trajectory with radius 1m and 20 poses
radius = 1
num_poses = 10
ground_truth_positons = []
for i in range(num_poses):
    x = radius * np.cos(i * 2 * np.pi / num_poses)
    y = radius * np.sin(i * 2 * np.pi / num_poses)
    theta = 0
    ground_truth_positons.append([x, y, theta])



# Odometry measurements with noise
odometry = []
noise_std = [0.01, 0.01, np.deg2rad(0.2)]  # [x_std, y_std, theta_std]
odometry_noise = np.diag([std**2 for std in noise_std])  # Covariance matrix
information = np.linalg.inv(odometry_noise)  # Information matrix (inverse of covariance)

# Generate Noisy Odometry Measurements (Increased error along the trajectory)


for i in range(len(ground_truth_positons) - 1):
    if i == 0:
        odometry.append(ground_truth_positons[i])
        continue
    else:
        # Compute the relative transformation between previous and current pose
        relative_transformation = relative_transform(ground_truth_positons[i], ground_truth_positons[i + 1])
        # Add this transformation + noise between previous odometry and current odometry
        noisy_pose = odometry[i-1] + np.random.multivariate_normal(relative_transformation, odometry_noise)
        odometry.append(noisy_pose)



information = np.eye(3) * 10

## Add Vertices to the graph
for i, pose in enumerate(odometry):
    if i == 0:
        graph_optimizer.add_pose_vertex_2D(i, pose, fixed=True)
    else:
        graph_optimizer.add_pose_vertex_2D(i, pose)

## Add Edges to the graph
for i in range(len(odometry) - 1):
    graph_optimizer.add_odometry_edge_2D(i,i+1, information)


## Add the loop closure edge between the first and last pose

graph_optimizer.add_edge_between_poses_2D(0, len(odometry) - 1, relative_transform(ground_truth_positons[0], ground_truth_positons[-1]), np.eye(3)*1000)


## Optimize the graph
graph_optimizer.optimize(10)

## Get the optimized poses
optimized_poses = graph_optimizer.get_all_poses_2D()

print("Optimized Poses", optimized_poses)
print("Odometry",odometry)





#Plot the ground truth trajectory and the noisy odometry measurements
plt.figure()
plt.plot([pose[0] for pose in ground_truth_positons], [pose[1] for pose in ground_truth_positons], 'g-', label='Ground Truth')
plt.plot([pose[0] for pose in odometry], [pose[1] for pose in odometry], 'r-', label='Odometry')
plt.plot([pose[0] for pose in optimized_poses], [pose[1] for pose in optimized_poses], 'b-', label='Optimized')
plt.legend()
plt.title("Ground Truth Trajectory and Odometry Measurements")
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.show()



# Ground truth positions and noise parameters
ground_truth_positons = [
    [0, 0, 0],
    [1.0, 0.0, np.deg2rad(90)],
    [1.0, 1.0, np.deg2rad(180)],
    [0.0, 1.0, np.deg2rad(-90)],
    [0.0, 0.0, 0]
]

odometry = [
            [0, 0, 0],
            [1.0404914245326837, 0.032285132702715345, 1.548058646294197],
            [1.007675476476252, 0.8930941070656758, 3.0668693044078004],
            [-0.008188478039399927, 0.9558269451002708, 4.646574416535024],
            [-0.03589007133357165, 0.020891631753120143, 6.274558744445183],
            ]

# Create a graph optimizer
graph_optimizer = GraphOptimizer()

for i, pose in enumerate(odometry):
    if i == 0:
        graph_optimizer.add_pose_vertex_2D(i, pose, fixed=True)
    else:
        graph_optimizer.add_pose_vertex_2D(i, pose)


information = np.eye(3)*10
# Add odometry edges
for i in range(len(odometry) - 1):
    graph_optimizer.add_odometry_edge_2D(i, i + 1, information)

graph_optimizer.add_edge_between_poses_2D(0, len(odometry) - 1, relative_transform(ground_truth_positons[0], ground_truth_positons[-1]), np.eye(3)*1000)

# Optimize the graph
graph_optimizer.optimize()

# Get the optimized poses
optimized_poses = graph_optimizer.get_all_poses_2D()


# Plot the Ground Truth, Noisy Odometry, and Optimized Poses
ground_truth_x = [pose[0] for pose in ground_truth_positons]
ground_truth_y = [pose[1] for pose in ground_truth_positons]
odometry_x = [pose[0] for pose in odometry]
odometry_y = [pose[1] for pose in odometry]
optimized_x = [pose[0] for pose in optimized_poses]
optimized_y = [pose[1] for pose in optimized_poses]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(ground_truth_x, ground_truth_y, label="Ground Truth", marker='o', linestyle='-', color='g')
plt.plot(odometry_x, odometry_y, label="Noisy Odometry", marker='x', linestyle='--', color='r')
plt.plot(optimized_x, optimized_y, label="Optimized", marker='s', linestyle='-', color='b')

# Labels and title
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Pose Graph Optimization: Ground Truth vs Noisy Odometry vs Optimized Poses")
plt.legend()

# Show plot
plt.grid(True)
plt.axis('equal')
plt.show()
