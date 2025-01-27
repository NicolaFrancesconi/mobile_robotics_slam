import os
import numpy as np

def save_datass( points , filename):
    """
    Save the map to a file
    :param map: Map to save
    :param filename: File to save to
    """
    path = os.path.abspath( __file__)
    print("Path: ", path)
    file_location_subfolders = 3 #Number of folder to go up to reach root of package
    for _ in range(file_location_subfolders):
        path = os.path.dirname(path)

    save_path = os.path.join(path, "trajectory_data")
    
    np.savetxt(os.path.join(save_path, filename), points)

    print("Data saved to: ", os.path.join(save_path, filename))