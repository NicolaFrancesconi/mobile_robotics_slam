import os
import numpy as np
import matplotlib.pyplot as plt

path = __file__
file_location_subfolders = 3 #Number of folder to go up to reach root of package
for _ in range(file_location_subfolders):
    path = os.path.dirname(path)

package_dir = path

# Open the txt files with the ground truth and the estimated trajectory
estimated_trajectory = np.loadtxt(os.path.join(package_dir, "trajectory_data", "robot_optimized.txt"))
odometry_trajectory = np.loadtxt(os.path.join(package_dir, "trajectory_data", "odom_trajectory.txt"))
icp_trajectory = np.loadtxt(os.path.join(package_dir, "trajectory_data", "icp_trajectory.txt"))


#Look at different tests how much the end position of the robot after a path is close to the origin

estimated_final_pose = estimated_trajectory[-1]
estimated_position_error = np.linalg.norm(estimated_final_pose[:2])
estimated_orientation_error = np.abs((0 - estimated_final_pose[2] + np.pi) % (2 * np.pi) - np.pi)
estimated_orientation_error = np.rad2deg(estimated_orientation_error)

odometry_final_pose = odometry_trajectory[-1]
odometry_position_error = np.linalg.norm(odometry_final_pose[:2])
odometry_orientation_error = np.abs((0 - odometry_final_pose[2] + np.pi) % (2 * np.pi) - np.pi)
odometry_orientation_error = np.rad2deg(odometry_orientation_error)

icp_final_pose = icp_trajectory[-1]
icp_position_error = np.linalg.norm(icp_final_pose[:2])
icp_orientation_error = np.abs((0 - icp_final_pose[2] + np.pi) % (2 * np.pi) - np.pi)
icp_orientation_error = np.rad2deg(icp_orientation_error)

print(f"[ICP Errors] Position: {icp_position_error} \t Orientation: {icp_orientation_error}")
print(f"[Odom Errors] Position: {odometry_position_error} \t Orientation: {odometry_orientation_error}")
print(f"[SLAM Errors] Position: {estimated_position_error} \t Orientation: {estimated_orientation_error}")

# #Plot the real, estimated and odometry trajectories
# plt.figure()
# plt.scatter(0,0, s=1, color="black")
# plt.plot(estimated_trajectory[:,0], estimated_trajectory[:,1], label="Graph SLAM", color="blue", linewidth=1)
# plt.plot(odometry_trajectory[:,0], odometry_trajectory[:,1], label="Odom", color="orange", linewidth=1)
# plt.plot(icp_trajectory[:,0], icp_trajectory[:,1], label="ICP", color="green", linewidth=1)
# plt.legend()
# plt.title("Trajectory Comparison")
# plt.xlabel("X [m]")
# plt.ylabel("Y [m]")
# plt.grid()
# plt.axis("equal")
# plt.show()







print("\\begin{table}[ht]")
print("\t\\centering")
print("\t\setlength\extrarowheight{2pt}")  # Needs \usepackage{array}
print("\t\\begin{tabular}{|c|c|c|c|}")
print("\t\t\hline")
print("\t\t  & Mean & Std & Max \\\\")
print("\t\\end{tabular}")
print("\t\\caption{Trajectory Error Analysis}")
print("\\end{table}")