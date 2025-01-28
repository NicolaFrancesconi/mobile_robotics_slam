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
icp_trajectory = np.loadtxt(os.path.join(package_dir, "trajectory_data", "icp_trajectory.txt"))

# Calculate the absolute trajectory error
def calculate_ATE_and_AOE(real_trajectory, estimated_trajectory):
    # Calculate the absolute trajectory error and the absolute orientation error
    ATE = 0
    AOE = 0
    position_error = []
    orientation_error = []
    for real_pose, estimate_pose in zip(real_trajectory, estimated_trajectory):
        pose_error = np.linalg.norm(real_pose[:2] - estimate_pose[:2])
        angle_error = np.abs((real_pose[2] - estimate_pose[2] + np.pi) % (2 * np.pi) - np.pi)
        angle_error = np.rad2deg(angle_error)
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

ATE, AOE,icp_position_error, icp_orientation_error = calculate_ATE_and_AOE(real_trajectory, icp_trajectory)

#Plot the real, estimated and odometry trajectories
plt.figure()
plt.plot(real_trajectory[:,0], real_trajectory[:,1], label="Real", linestyle="--", linewidth=2.5, color="red")
plt.plot(estimated_trajectory[:,0], estimated_trajectory[:,1], label="Graph SLAM", color="blue", linewidth=1)
plt.plot(odometry_trajectory[:,0], odometry_trajectory[:,1], label="Odom", color="orange", linewidth=1)
plt.plot(icp_trajectory[:,0], icp_trajectory[:,1], label="ICP", color="green", linewidth=1)
plt.legend()
plt.title("Trajectory Comparison")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.grid()
plt.axis("equal")

# Plot the orientation error and position error in the same figure, one above the other
fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(10, 8))

zeros_error = np.zeros(len(SLAM_orientation_error))
# Plot the orientation error
mean_orientation_error = np.mean(SLAM_orientation_error)
for i in range(len(SLAM_orientation_error)):
    ax1.plot([i, i], [zeros_error[i], SLAM_orientation_error[i]], color="blue", alpha=0.5)
    ax1.scatter( i,SLAM_orientation_error[i], label='Orientation Error' if i == 0 else "", color="blue")
ax1.axhline(y=mean_orientation_error, color='r', linestyle='--', label="Mean Error")
ax1.set_xlabel("Pose Number")
ax1.set_ylabel("Orientation Error [deg]")
ax1.set_title("Orientation Error SLAM")
ax1.grid()
ax1.legend()

# Plot the position error
mean_position_error = np.mean(SLAM_position_error)
for i in range(len(SLAM_position_error)):
    ax2.plot([i, i], [zeros_error[i], SLAM_position_error[i]], color="blue", alpha=0.5)
    ax2.scatter(i, SLAM_position_error[i], label='Position Error' if i == 0 else "", color="blue")
ax2.axhline(y=mean_position_error, color='r', linestyle='--', label="Mean Error")
ax2.set_xlabel("Pose Number")
ax2.set_ylabel("Position Error [m]")
ax2.set_title("Position Error SLAM")
ax2.grid()
ax2.legend()

plt.tight_layout()
plt.show()

#Plot the orientation error and position error for the odometry trajectory
fig2, (ax2, ax1) = plt.subplots(2, 1, figsize=(10, 8))
mean_orientation_error = np.mean(odometry_orientation_error)
for i in range(len(odometry_orientation_error)):
    ax1.plot([i, i], [zeros_error[i], odometry_orientation_error[i]], color="orange", alpha=0.5)
    ax1.scatter( i,odometry_orientation_error[i], label='Orientation Error' if i == 0 else "", color="orange")
ax1.axhline(y=mean_orientation_error, color='r', linestyle='--', label="Mean Error")
ax1.set_xlabel("Pose Number")
ax1.set_ylabel("Orientation Error [deg]")
ax1.set_title("Orientation Error Odometry")
ax1.grid()
ax1.legend()

mean_position_error = np.mean(odometry_position_error)
for i in range(len(odometry_position_error)):
    ax2.plot([i, i], [zeros_error[i], odometry_position_error[i]], color="orange", alpha=0.5)
    ax2.scatter(i, odometry_position_error[i], label='Position Error' if i == 0 else "", color="orange")
ax2.axhline(y=mean_position_error, color='r', linestyle='--', label="Mean Error")
ax2.set_xlabel("Pose Number")
ax2.set_ylabel("Position Error [m]")
ax2.set_title("Position Error Odometry")
ax2.grid()
ax2.legend()

plt.tight_layout()
plt.show()

#Plot the orientation error and position error for the ICP trajectory
fig3, (ax2, ax1) = plt.subplots(2, 1, figsize=(10, 8))
mean_orientation_error = np.mean(icp_orientation_error)
for i in range(len(icp_orientation_error)):
    ax1.plot([i, i], [zeros_error[i], icp_orientation_error[i]], color="green", alpha=0.5)
    ax1.scatter( i,icp_orientation_error[i], label='Orientation Error' if i == 0 else "", color="green")
ax1.axhline(y=mean_orientation_error, color='r', linestyle='--', label="Mean Error")
ax1.set_xlabel("Pose Number")
ax1.set_ylabel("Orientation Error [deg]")
ax1.set_title("Orientation Error ICP")
ax1.grid()
ax1.legend()

mean_position_error = np.mean(icp_position_error)
for i in range(len(icp_position_error)):
    ax2.plot([i, i], [zeros_error[i], icp_position_error[i]], color="green", alpha=0.5)
    ax2.scatter(i, icp_position_error[i], label='Position Error' if i == 0 else "", color="green")
ax2.axhline(y=mean_position_error, color='r', linestyle='--', label="Mean Error")
ax2.set_xlabel("Pose Number")
ax2.set_ylabel("Position Error [m]")
ax2.set_title("Position Error ICP")
ax2.grid()
ax2.legend()

plt.tight_layout()
plt.show()


# Calculate the mean and standard deviation of the position and orientation error
mean_position_error = np.mean(SLAM_position_error)
std_position_error = np.std(SLAM_position_error- mean_position_error)
mean_orientation_error = np.mean(SLAM_orientation_error)
std_orientation_error = np.std(SLAM_orientation_error - mean_orientation_error)

mean_odometry_position_error = np.mean(odometry_position_error)
std_odometry_position_error = np.std(odometry_position_error- mean_odometry_position_error)
mean_odometry_orientation_error = np.mean(odometry_orientation_error)
std_odometry_orientation_error = np.std(odometry_orientation_error - mean_odometry_orientation_error)

mean_icp_position_error = np.mean(icp_position_error)
std_icp_position_error = np.std(icp_position_error- mean_icp_position_error)
mean_icp_orientation_error = np.mean(icp_orientation_error)
std_icp_orientation_error = np.std(icp_orientation_error - mean_icp_orientation_error)



print("\\begin{table}[ht]")
print("\t\\centering")
print("\t\setlength\extrarowheight{2pt}")  # Needs \usepackage{array}
print("\t\\begin{tabular}{|c|c|c|c|}")
print("\t\t\hline")
print("\t\t  & Mean & Std & Max \\\\")
print("\t\t\hline")
print(f"\t\tSLAM Position Error & {mean_position_error:.5f} m & {std_position_error:.5f} m & {max(SLAM_position_error):.5f} m \\\\")
print("\t\t\hline")
print(f"\t\tOdometry Position Error & {mean_odometry_position_error:.5f} m & {std_odometry_position_error:.5f} m & {max(odometry_position_error):.5f} m \\\\")
print("\t\t\hline")
print(f"\t\tICP Position Error & {mean_icp_position_error:.5f} m & {std_icp_position_error:.5f} m & {max(icp_position_error):.5f} m \\\\")
print("\t\t\hline")
print(f"\t\tSLAM Orientation Error & {mean_orientation_error:.5f} deg & {std_orientation_error:.5f} deg & {max(SLAM_orientation_error):.5f} deg \\\\")
print("\t\t\hline")
print(f"\t\tOdometry Orientation Error & {mean_odometry_orientation_error:.5f} deg & {std_odometry_orientation_error:.5f} deg & {max(odometry_orientation_error):.5f} deg \\\\")
print("\t\t\hline")
print(f"\t\tICP Orientation Error & {mean_icp_orientation_error:.5f} deg & {std_icp_orientation_error:.5f} deg & {max(icp_orientation_error):.5f} deg \\\\")
print("\t\t\hline")
print("\t\\end{tabular}")
print("\t\\caption{Trajectory Error Analysis}")
print("\\end{table}")





