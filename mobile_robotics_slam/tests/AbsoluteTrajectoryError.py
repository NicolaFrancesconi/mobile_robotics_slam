import os
import numpy as np
import matplotlib.pyplot as plt

path = __file__
file_location_subfolders = 3 #Number of folder to go up to reach root of package
for _ in range(file_location_subfolders):
    path = os.path.dirname(path)

package_dir = path

# Open the txt files with the ground truth and the estimated trajectory
real_trajectory = np.loadtxt(os.path.join(package_dir, "trajectory_data", "real_trajectory.txt"))
estimated_trajectory = np.loadtxt(os.path.join(package_dir, "trajectory_data", "robot_optimized.txt"))
odometry_trajectory = np.loadtxt(os.path.join(package_dir, "trajectory_data", "odom_trajectory.txt"))

# Calculate the absolute trajectory error
def calculate_ATE_and_AOE(real_trajectory, estimated_trajectory):
    # Calculate the absolute trajectory error and the absolute orientation error
    ATE = 0
    AOE = 0
    position_error = []
    orientation_error = []
    for real_pose, estimate_pose in zip(real_trajectory, estimated_trajectory):
        pose_error = np.linalg.norm(real_pose - estimate_pose)
        angle_error = np.abs((real_pose[2] - estimate_pose[2] + np.pi) % (2 * np.pi) - np.pi)
        ATE += pose_error
        position_error.append(pose_error)
        AOE += angle_error
        orientation_error.append(angle_error)
    ATE /= len(real_trajectory)
    AOE /= len(real_trajectory)
    return ATE, AOE, position_error, orientation_error

# Calculate the absolute trajectory error and the absolute orientation error
ATE, AOE, SLAM_position_error, SLAM_orientation_error = calculate_ATE_and_AOE(real_trajectory, estimated_trajectory)
print("Absolute Trajectory Error: ", ATE)
print("Absolute Orientation Error: ", AOE)

# Calculate the absolute trajectory error and the absolute orientation error for the odometry trajectory
ATE, AOE,odometry_position_error, odometry_orientation_error = calculate_ATE_and_AOE(real_trajectory, odometry_trajectory)
print("Absolute Trajectory Error Odometry: ", ATE)
print("Absolute Orientation Error Odometry: ", AOE)

#Plot the real, estimated and odometry trajectories
plt.figure()
plt.plot(real_trajectory[:,0], real_trajectory[:,1], label="Real", linestyle="--", linewidth=2.5, color="red")
plt.plot(estimated_trajectory[:,0], estimated_trajectory[:,1], label="Graph SLAM", color="blue", linewidth=1)
plt.plot(odometry_trajectory[:,0], odometry_trajectory[:,1], label="Odom", color="orange", linewidth=1)

plt.legend()
plt.title("Trajectory Comparison")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.grid()
plt.axis("equal")
plt.show()

# Plot the orientation error
plt.figure()
plt.plot(SLAM_orientation_error, label="Error", color="blue")
plt.xlabel("Pose Number")
plt.ylabel("Orientation Error [rad]")
plt.title("Orientation Error Comparison")
plt.grid()
plt.legend()
plt.show()

# Plot the position error
plt.figure()
plt.plot(SLAM_position_error, label="Error", color="blue")
plt.xlabel("Pose Number")
plt.ylabel("Position Error [m]")
plt.title("Position Error Comparison")
plt.grid()
plt.legend()
plt.show()


