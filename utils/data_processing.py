import numpy as np
import os

def load_pore_matrices(file_paths: list[str]):
    """
    Loads .npy files and reshapes it into a 30x30x30 matrix representing pore occupancy.

    Parameters:
    - file_path (lst[str]): List of paths to .npy files.

    Returns:
    - dict[np.ndarray]: simulation data
    - dict[np.ndarray]: 3D numpy array with shape (30, 30, 30) containing 0s and 1s.
    - dict[int]: density factors
    """
    
    data_dict = {}
    grid_dict = {}
    density_dict = {}

    for i, file in enumerate(file_paths):
        _, index, density = file.split("\\")[-1].replace(".npy", "").split("_")
        data = np.load(file)
        occupancy = data[:, 3]
        pore_matrix = occupancy.reshape((30, 30, 30))
        
        data_dict[index] = data
        grid_dict[index] = pore_matrix
        density_dict[index] = float(density)
    
    return data_dict, grid_dict, density_dict
