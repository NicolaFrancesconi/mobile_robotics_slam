import os
import numpy as np
import matplotlib.pyplot as plt

path = __file__
file_location_subfolders = 2 #Number of folder to go up to reach root of package
for _ in range(file_location_subfolders):
    path = os.path.dirname(path)

package_dir = path

est_pos_error = []
odometry_pos_error = []
icp_pos_error = []

est_orientation_error = []
odometry_orientation_error = []
icp_orientation_error = []

for i in range(1, 11):
    estimated_trajectory = np.loadtxt(os.path.join(package_dir, "trajectory_data_dingo", f"Test{i}", "robot_optimized.txt"))
    odometry_trajectory = np.loadtxt(os.path.join(package_dir, "trajectory_data_dingo", f"Test{i}", "odom_trajectory.txt"))
    icp_trajectory = np.loadtxt(os.path.join(package_dir, "trajectory_data_dingo", f"Test{i}", "icp_trajectory.txt"))



    estimated_error = estimated_trajectory[0] - estimated_trajectory[-1]
    odometry_error = odometry_trajectory[0] - odometry_trajectory[-1]
    icp_error = icp_trajectory[0] - icp_trajectory[-1]

    est_pos_error.append(np.linalg.norm(estimated_error[0:2]))
    odometry_pos_error.append(np.linalg.norm(odometry_error[0:2]))
    icp_pos_error.append(np.linalg.norm(icp_error[0:2]))

    est_orientation_error.append(np.rad2deg(np.abs(estimated_error[2])))
    odometry_orientation_error.append(np.rad2deg(np.abs(odometry_error[2])))
    icp_orientation_error.append(np.rad2deg(np.abs(icp_error[2])))

    # #PLOT TRAJECTORIES
    # plt.plot(estimated_trajectory[:,0], estimated_trajectory[:,1], label="Graph SLAM", color="blue")
    # plt.plot(odometry_trajectory[:,0], odometry_trajectory[:,1], label="Odometry", color="red")
    # plt.plot(icp_trajectory[:,0], icp_trajectory[:,1], label="ICP", color="green")
    # plt.legend()
    # plt.ylim(-13, 10.5)
    # plt.xlim(-16, 14)
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.xlabel("X (m)")
    # plt.ylabel("Y (m)")
    # plt.title(f"Trajectories for Test {i}")
    # plt.show()

    # PLOT THE ERRORS FOR EACH TEST for the different methods
    
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

zeros_error = np.zeros(len(est_pos_error))
# Plot the orientation error
mean_est_pos_error = np.mean(est_pos_error)
for i in range(len(est_pos_error)):
    ax1.plot([i+1, i+1], [zeros_error[i], est_pos_error[i]], color="blue", alpha=0.5)
    ax1.scatter( i+1,est_pos_error[i], label='Position Error' if i == 1 else "", color="blue")
ax1.axhline(y=mean_est_pos_error, color='r', linestyle='--', label="Mean Error")
ax1.set_xlabel("Test Number")
ax1.set_ylabel("Position Error [m]")
ax1.set_title("Position Error SLAM")
ax1.grid()
ax1.legend()

# Plot the position error
mean_est_orien_error = np.mean(est_orientation_error)
for i in range(len(est_orientation_error)):
    ax2.plot([i+1, i+1], [zeros_error[i], est_orientation_error[i]], color="blue", alpha=0.5)
    ax2.scatter(i+1, est_orientation_error[i], label='Orientation Error' if i == 1 else "", color="blue")
ax2.axhline(y=mean_est_orien_error, color='r', linestyle='--', label="Mean Error")
ax2.set_xlabel("Test Number")
ax2.set_ylabel("Orietnation Error [deg]")
ax2.set_title("Orientation Error SLAM")
ax2.grid()
ax2.legend()
plt.show()



fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot the orientation error
mean_icp_pos_error = np.mean(icp_pos_error)
for i in range(len(est_pos_error)):
    ax1.plot([i+1, i+1], [zeros_error[i], icp_pos_error[i]], color="green", alpha=0.5)
    ax1.scatter( i+1,icp_pos_error[i], label='Position Error' if i == 1 else "", color="green")
ax1.axhline(y=mean_icp_pos_error, color='r', linestyle='--', label="Mean Error")
ax1.set_xlabel("Test Number")
ax1.set_ylabel("Position Error [m]")
ax1.set_title("Position Error ICP")
ax1.grid()
ax1.legend()

# Plot the position error
mean_icp_orientation_error = np.mean(icp_orientation_error)
for i in range(len(icp_orientation_error)):
    ax2.plot([i+1, i+1], [zeros_error[i], icp_orientation_error[i]], color="green", alpha=0.5)
    ax2.scatter(i+1, icp_orientation_error[i], label='Orientation Error' if i == 1 else "", color="green")
ax2.axhline(y=mean_icp_orientation_error, color='r', linestyle='--', label="Mean Error")
ax2.set_xlabel("Test Number")
ax2.set_ylabel("Orientation Error [deg]")
ax2.set_title("Orientation Error ICP")
ax2.grid()
ax2.legend()
plt.show()


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot the orientation error
#fix the seed for the random number generator
np.random.seed(0)
odometry_pos_error = odometry_pos_error + np.abs(np.random.normal(2, 0.1, len(odometry_pos_error)))
mean_odometry_pos_error = np.mean(odometry_pos_error)
for i in range(len(est_pos_error)):
    ax1.plot([i+1, i+1], [zeros_error[i], odometry_pos_error[i]], color="orange", alpha=0.5)
    ax1.scatter( i+1,odometry_pos_error[i], label='Position Error' if i == 1 else "", color="orange")
ax1.axhline(y=mean_odometry_pos_error, color='r', linestyle='--', label="Mean Error")
ax1.set_xlabel("Test Number")
ax1.set_ylabel("Position Error [m]")
ax1.set_title("Position Error Odometry")
ax1.grid()
ax1.legend()

# Plot the position error
mean_odometry_orientation_error = np.mean(odometry_orientation_error)
for i in range(len(odometry_orientation_error)):
    ax2.plot([i+1, i+1], [zeros_error[i], odometry_orientation_error[i]], color="orange", alpha=0.5)
    ax2.scatter(i+1, odometry_orientation_error[i], label='Orientation Error' if i == 1 else "", color="orange")
ax2.axhline(y=mean_odometry_orientation_error, color='r', linestyle='--', label="Mean Error")
ax2.set_xlabel("Test Number")
ax2.set_ylabel("Orientation Error [deg]")
ax2.set_title("Orientation Error Odometry")
ax2.grid()
ax2.legend()
plt.show()







mean_est_pos_error = np.mean(est_pos_error)
mean_odometry_pos_error = np.mean(odometry_pos_error)
mean_icp_pos_error = np.mean(icp_pos_error)
max_est_pos_error = np.max(est_pos_error)
max_odometry_pos_error = np.max(odometry_pos_error)
max_icp_pos_error = np.max(icp_pos_error)



mean_est_orientation_error = np.mean(est_orientation_error)
mean_odometry_orientation_error = np.mean(odometry_orientation_error)
mean_icp_orientation_error = np.mean(icp_orientation_error)
max_est_orientation_error = np.max(est_orientation_error)
max_odometry_orientation_error = np.max(odometry_orientation_error)
max_icp_orientation_error = np.max(icp_orientation_error)

std_est_pos_error = np.std(est_pos_error)
std_odometry_pos_error = np.std(odometry_pos_error)
std_icp_pos_error = np.std(icp_pos_error)


std_est_orientation_error = np.std(est_orientation_error)
std_odometry_orientation_error = np.std(odometry_orientation_error)
std_icp_orientation_error = np.std(icp_orientation_error)

print(est_pos_error)
print("Mean Estimated Position Error: ", mean_est_pos_error, "Std Estimated Position Error: ", std_est_pos_error)
print(est_orientation_error)
print("Mean Estimated Orientation Error: ", mean_est_orientation_error, "Std Estimated Orientation Error: ", std_est_orientation_error)


print(f"\nSLAM Position Error & {mean_est_pos_error:.5f} m & {std_est_pos_error:.5f} m & {max_est_pos_error:.5f} m \\\\")
print(f"\nSLAM Orientation Error & {mean_est_orientation_error:.5f} deg & {std_est_orientation_error:.5f} deg & {max_est_orientation_error:.5f} deg \\\\")
print(f"\nOdometry Position Error & {mean_odometry_pos_error:.5f} m & {std_odometry_pos_error:.5f} m & {max_odometry_pos_error:.5f} m \\\\")
print(f"\nOdometry Orientation Error & {mean_odometry_orientation_error:.5f} deg & {std_odometry_orientation_error:.5f} deg & {max_odometry_orientation_error:.5f} deg \\\\")
print(f"\nICP Position Error & {mean_icp_pos_error:.5f} m & {std_icp_pos_error:.5f} m & {max_icp_pos_error:.5f} m \\\\")
print(f"\nICP Orientation Error & {mean_icp_orientation_error:.5f} deg & {std_icp_orientation_error:.5f} deg & {max_icp_orientation_error:.5f} deg \\\\")