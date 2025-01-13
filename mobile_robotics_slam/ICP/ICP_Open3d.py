# Original code: https://github.com/ClayFlannigan/icp
# Modified to reject pairs that have greater distance than the specified threshold
# Add covariance check

import numpy as np
import matplotlib.pyplot as plt
import time
import open3d as o3d

def homogeneous_transform(poseB, poseA):
    """Compute the homogeneous transformation matrix from pose A to pose B
    Input:
    poseA: 3x1 numpy array of the [x, y, theta] pose in the A frame
    poseB: 3x1 numpy array of the [x, y, theta] pose in the B frame
    Output:
    T: 4x4 numpy array of the homogeneous transformation matrix from pose A to pose B"""
    T = np.eye(4)
    T[0:2, 3] = poseB[0:2] - poseA[0:2]
    T[0, 0] = np.cos(poseB[2])
    T[0, 1] = -np.sin(poseB[2])
    T[1, 0] = np.sin(poseB[2])
    T[1, 1] = np.cos(poseB[2])
    return T

    
def icp(previous_points,  current_points, init_homog_estimate=None, associate_max_distance=5):
    """Perform ICP with Open3D
    Input:
    current_scan: Nx3 numpy array of points [x, y, z=0] in the current scan
    previous_scan: Nx3 numpy array of points [x, y, z=0] in the previous scan
    init_homog_estimate: 4x4 numpy array of the initial homogeneous transformation from current to previous scan
    associate_max_distance: Maximum distance for point association
    Output:
    transformation: 4x4 numpy array of the estimated transformation from current to previous scan"""
    initial_guess = np.eye(4)
    threshold = associate_max_distance  # Maximum correspondence distance (adjust based on your data)
    if init_homog_estimate is not None:
        initial_guess[:2, :2] = init_homog_estimate[:2, :2]
        initial_guess[:2, 3] = init_homog_estimate[:2, 2]
        print("Initial Guess:\n", initial_guess)

    #Set Verbosity
    #o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(previous_points)
    


    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(current_points)

    #Create Voxels
    # voxel_size = 0.02
    # pcd1_down = pcd1.voxel_down_sample(voxel_size)
    # pcd2_down = pcd2.voxel_down_sample(voxel_size)

    

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
    relative_fitness=1e-6,
    max_iteration=30,
    relative_rmse=1e-6,
    )

    # Perform ICP registration
    
    icp_result = o3d.pipelines.registration.registration_icp(
        pcd2,
        pcd1,
        threshold,
        initial_guess,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=criteria
        
        )
    T = icp_result.transformation

    
    print("T:\n", T)
    current_homog_points = np.hstack((current_points, np.ones((current_points.shape[0], 1))))
    previous_homog_points = np.hstack((previous_points, np.ones((previous_points.shape[0], 1))))

    points_transformed_ICP = current_homog_points@T.T
    points_transformed_initial = current_homog_points@initial_guess.T

    # plt.figure()
    # plt.scatter(current_points[:, 0], current_points[:, 1], c='r', label='Source', s=10)
    # plt.scatter(previous_points[:, 0], previous_points[:, 1], c='b', label='Target', s=10)
    # plt.scatter(points_transformed_ICP[:, 0], points_transformed_ICP[:, 1], c='g', label='Source Transformed Using ICP', s=15)
    # plt.scatter(points_transformed_initial[:, 0], points_transformed_initial[:, 1], c='y', label='Source Transformed Using Initial Guess', s=15)
    # plt.axis('equal')
    # plt.legend()
    # plt.show()
    # # print("Fit Error: ", icp_result.fitness)
    # # print("Inlier RMSE: ", icp_result.inlier_rmse)
    T_2d = np.eye(3)
    T_2d[0:2, 0:2] = T[0:2, 0:2]
    T_2d[0:2, 2] = T[0:2, 3]
    #T_2d[0:2, 2] = init_homog_estimate[0:2, 2] 

    print("Transformation Matrix:\n", T_2d)
    #print("Inverse Transformation Matrix:\n", np.linalg.inv(T_2d))
    return init_homog_estimate

    return T_2d


def points_3d_from_scan_and_pose(scan, robot_estimated_pose=np.zeros(3), max_range=30, downsample=1):
        angles = np.linspace(-np.pi, np.pi, len(scan))
        scan = np.array(scan)
        ranges = scan[scan < max_range][::downsample]
        angles = angles[scan < max_range][::downsample]
        x = ranges * np.cos(angles + robot_estimated_pose[2]) + robot_estimated_pose[0]
        y = ranges * np.sin(angles + robot_estimated_pose[2]) + robot_estimated_pose[1]
        return np.vstack((x, y, np.zeros_like(x))).T


def compute_homo_transform( pose1, pose2):
        T1 = pose_to_transform(pose1)
        T2 = pose_to_transform(pose2)
        H = np.linalg.inv(T1)@T2
        distance = np.linalg.norm(H[0:2, 2])
        rotation = np.arctan2(H[1,0], H[0,0])

        return H, distance, rotation

def pose_to_transform(pose):
        """Given a pose [x,y,theta] it returns the Homogeneous
        transform T of the pose"""
        cos = np.cos(pose[2])
        sin = np.sin(pose[2])
        dx = pose[0]
        dy = pose[1]
        T = np.array([[cos, -sin, dx],
                      [sin, cos , dy],
                      [0  , 0   , 1 ]])
        return T



# rangesA = np.loadtxt('scan1.txt')
# rangesB = np.loadtxt('scan2.txt')

# # # start = time.time()

# poseA = np.array([0.0, 0.0, 0.0])
# poseB = np.array([0.0, 0.4, 0.2])

# # # # pointsA = points_3d_from_scan_and_pose(rangesA, poseA, downsample=20)
# # # # pointsB = points_3d_from_scan_and_pose(rangesB, poseB, downsample=20)
# # # # H_est = np.eye(3)

# # # # list_time = []
# # # # for i in range(30):
# # # #     temps  = time.time()
# # # #     T_est =icp(pointsA, pointsB)
# # # #     #print("Time for ICP:", time.time()- temps)
# # # #     list_time.append(time.time()- temps)

# # # # #print("Real Transformation:\n", H_est) 
# # # # print("Mean Time: ", np.mean(list_time))


# pointsA= points_3d_from_scan_and_pose(rangesA, downsample=20)
# pointsB= points_3d_from_scan_and_pose(rangesB, downsample=20)

# pointsA = np.array([[0., 0., 0.], [0., 1., 0.], [1., 1., 0.], [1., 0., 0.]])
# #Rotate and Translate points A to get points B
# rot = np.deg2rad(2)
# poseA = np.array([0.0, 0.0, 0.0])
# poseB = np.array([0.0, 0.1, rot])
# pointsB = np.dot(np.array([[np.cos(rot), -np.sin(rot), 0.], [np.sin(rot), np.cos(rot), 0.], [0., 0., 1.]]), pointsA.T).T + np.array([0.0, 0.1, 0.0])
# pointsB = pointsB + np.random.normal(0, 0.001, pointsB.shape)

# #pointsB = pointsA + np.array([0.0, 0.4, 0.0])
# H_est = compute_homo_transform(poseA, poseB)[0]

# list_time = []
# for i in range(1):
#     temps  = time.time()
#     T_est =icp(pointsA, pointsB, init_homog_estimate=H_est)
#     #print("Time for ICP:", time.time()- temps)
#     list_time.append(time.time()- temps)

# #print("Real Transformation:\n", H_est) 
# print("Mean Time: ", np.mean(list_time))

# pointsB_transformed = np.dot(T_est[0:3, 0:3].T, pointsB.T) - T_est[0:3, 3].reshape(3, 1)

# print("Time: ", time.time()-start)
# rotation = np.arctan2(T_est[1, 0], T_est[0, 0])

# print("Rotation E: ", -rotation)

# # Plot the scans and the transformed scans
# plt.figure()
# plt.scatter(pointsA[:, 0], pointsA[:, 1], c='r', label='Scan A')
# plt.scatter(pointsB[:, 0], pointsB[:, 1], c='b', label='Scan B')
# plt.scatter(pointsB_transformed[0, :], pointsB_transformed[1, :], c='g', label='Scan B Transformed')
# plt.axis('equal')
# plt.legend()
# plt.show()



