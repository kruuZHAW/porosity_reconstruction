import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

class PoreDataset(Dataset):
    def __init__(self, npy_files):
        self.npy_files = npy_files 

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
        file_path = self.npy_files[idx]
        data = np.load(file_path) 

        occupancy = data[:,3]
        grid = occupancy.reshape((30, 30, 30))
        
        # Extract density from filename
        density_factor = float(file_path.split("_")[-1].replace(".npy", ""))
        
        # Convert to tensors
        grid_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)  # Shape (1,30,30,30)
        density_tensor = torch.tensor([density_factor], dtype=torch.float32)

        return grid_tensor, density_tensor