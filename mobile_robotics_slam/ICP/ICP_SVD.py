# Original code: https://github.com/ClayFlannigan/icp
# Modified to reject pairs that have greater distance than the specified threshold
# Add covariance check

import numpy as np
from sklearn.neighbors import NearestNeighbors


def icp(source, target, init_transform=None, max_iterations=30, tolerance=1e-4, max_ass_distance=1.0, downsample=1, max_range=90):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Kxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
        max_ass_distance: Distance for Nearest Neigh Association
        downsample: downsample of the original point cloud
        max_range: max distance a of a point in the point cloud to be considered in ICP
    Output:
        T: final homogeneous transformation that maps source on to target
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert source.shape[1] == target.shape[1]

    m = source.shape[1]

    distA = np.linalg.norm(source, axis=1)
    distB = np.linalg.norm(target, axis=1)

    filteredA = source[distA < max_range]
    filteredB = target[distB < max_range]

    minLA = min(len(filteredA), len(source)//downsample)
    minLB = min(len(filteredB), len(target)//downsample)

    #Randomly Remove points to get the downsampled A and B
    shuffledA = np.random.permutation(filteredA)[:minLA]
    shuffledB = np.random.permutation(filteredB)[:minLB]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,shuffledA.shape[0]))
    dst = np.ones((m+1,shuffledB.shape[0]))
    src[:m,:] = np.copy(shuffledA.T)
    dst[:m,:] = np.copy(shuffledB.T)

    T_final = np.identity(m+1)
    # apply the initial transform estimation
    if init_transform is not None:
        assert init_transform.shape == (m+1, m+1)
        src = np.dot(init_transform, src)
        T_final = np.copy(init_transform)

    prev_error = 0.0

    for i in range(max_iterations):
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T) # Nearest neighbor search
        mask = distances < max_ass_distance
        mean_error = np.mean(distances[mask])
        delta_error = np.abs(prev_error - mean_error)
        prev_error = mean_error
        if delta_error < tolerance:
            print(f"Converged at iteration {i}")
            break
        
        indices = indices[mask] #Take Only Indices with distance lower than Max admissible
        T,_,_ = svd_transform(src[:m, mask].T, dst[:m,indices].T) #Compute Transformation
        T_final = T_final@T #Update Final Transformation adding the new relative T
        src = np.dot(T, src) #Apply Last transformation to the "already modified" source points

    cov = np.eye(3) #TODO: Evaluate Covariance of The allignement

    return T_final

def svd_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''
    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t

def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Kxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()



