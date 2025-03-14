import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def threshold_top_values(grid, density):
    """
    Transforms a 30x30x30 grid into a binary grid by keeping only the `nb_ones` highest values.
    
    Computes nb_ones based on the formula:
    Density = 0.0882 * (Number of Ones)^0.2632
    
    Parameters:
    - grid: numpy array of shape (30, 30, 30), with continuous values (e.g., probabilities)
    - density: float, density value used to compute `nb_ones`
    
    Returns:
    - binary_grid: numpy array of shape (30, 30, 30), where only the top `nb_ones` values are 1, others are 0.
    """
    # Compute the number of ones to keep based on the given formula
    nb_ones = int((density/0.0882)**(1/0.2632))

    # Flatten grid and get the threshold value for the top `nb_ones` values
    sorted_values = np.sort(grid.flatten())[::-1]  # Sort in descending order
    threshold = sorted_values[nb_ones - 1] if nb_ones > 0 else 1  # Ensure valid threshold

    # Create binary grid by thresholding
    binary_grid = (grid >= threshold).astype(int)

    return binary_grid

def intersection_over_union(grid1, grid2, density, epsilon=1e-8):
    """
    Computes the Intersection over Union (IoU) between two 3D binary grids.

    Args:
        grid1 (torch.Tensor): First grid (ground truth).
        grid2 (torch.Tensor): Second grid (prediction).
        epsilon (float): Small value to avoid division by zero.

    Returns:
        iou (torch.Tensor): The IoU score for each batch element.
    """
     # Transform generation into 0 and ones
    with torch.no_grad():
        binary_grid2 = threshold_top_values(grid2.squeeze().cpu().numpy(), density)

    # Compute intersection and union
    intersection = torch.sum(grid1 * binary_grid2, dim=(0, 1, 2))
    union = torch.sum(grid1 + binary_grid2, dim=(0, 1, 2)) - intersection  

    # Compute IoU
    iou = intersection / (union + epsilon)  # Avoid division by zero

    return iou

def misclassified_ones(grid1, grid2, density):
    """
    Computes the number of misclassified ones (False Negatives + False Positives) in a 3D grid.

    Args:
        grid1 (torch.Tensor): Ground truth binary grid of shape (D, H, W).
        grid2 (torch.Tensor): Predicted probability grid of shape (D, H, W).
        density (int): Number of ones expected in the prediction.

    Returns:
        misclassified (int): Total misclassified ones (FN + FP).
        false_negatives (int): Count of false negatives.
        false_positives (int): Count of false positives.
    """
    # Convert predicted grid2 into binary format based on top-k thresholding
    with torch.no_grad():
        binary_grid2 = threshold_top_values(grid2.squeeze().cpu().numpy(), density)

    # Convert to PyTorch tensor
    binary_grid2 = torch.tensor(binary_grid2, dtype=torch.float32, device=grid1.device)

    # False Negatives (FN): Ones in ground truth missing in prediction
    false_negatives = torch.sum(grid1 * (1 - binary_grid2))

    # False Positives (FP): Ones in prediction that should be zero
    false_positives = torch.sum((1 - grid1) * binary_grid2)

    # Total misclassified ones
    misclassified = false_negatives + false_positives

    return misclassified.item(), false_negatives.item(), false_positives.item()


def plot_pore_reconstruction(grid1, grid2, index1, index2, density):
    """
    Plots two 3D scatter plots side by side for two different 30x30x30 grids representing pore structures.
    
    Parameters:
    - grid1, grid2: numpy arrays of shape (30, 30, 30), where values are 0 (solid) or 1 (pore)
    - index1, index2: Identifiers for the grids
    - densities_dict: Dictionary containing density values for each grid
    """
    
    # Transform generation into 0 and ones
    with torch.no_grad():
        binary_grid = threshold_top_values(grid2.squeeze().cpu().numpy(), density)

    # Get the (x, y, z) coordinates where occupancy is 1 (pores)
    x1, y1, z1 = np.where(grid1 == 1)
    x2, y2, z2 = np.where(binary_grid == 1)

    # Normalize coordinates to be in the unit cube [0,1]
    x1, y1, z1 = x1 / 29, y1 / 29, z1 / 29
    x2, y2, z2 = x2 / 29, y2 / 29, z2 / 29

    # Create side-by-side 3D scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={'projection': '3d'})

    # Plot first grid
    axes[0].scatter(x1, y1, z1, c='blue', marker='o', alpha=0.5, s=5)
    axes[0].set_title(f"Pore Visualization: {index1}", fontsize = 18)
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].set_zlabel("Z")

    # Plot second grid
    axes[1].scatter(x2, y2, z2, c='red', marker='o', alpha=0.5, s=5)
    axes[1].set_title(f"Pore Visualization: {index2}", fontsize = 18)
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    axes[1].set_zlabel("Z")

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()
    

def generate_pore_grid(model, density_factor=0.3, random_sample=False):
    """
    Generates and visualizes a 3D pore structure using the trained VAE model.

    Parameters:
    - model: Trained VAE model
    - density_factor (float): The density value to condition the generation on
    - random_sample (bool): If True, samples a random latent vector
    """

    model.eval()  # Set to evaluation mode

    with torch.no_grad():
        # Sample a random latent vector or use mean latent representation
        latent_dim = model.latent_dim
        if random_sample:
            z = torch.randn(1, latent_dim)  # Random sampling from a standard gaussian
        else:
            z = torch.zeros(1, latent_dim)  # Use mean latent vector
            
        density_tensor = torch.tensor([[density_factor]], dtype=torch.float32)

        # Generate 3D pore grid
        generated_grid = model.decode(z, density_tensor).squeeze().cpu().numpy()

        # Convert to binary grid using model for number of ones
        binary_grid = threshold_top_values(generated_grid, density_factor)

        # Extract coordinates of pore locations (where grid == 1)
        x, y, z = np.where(binary_grid == 1)

        # Plot 3D scatter of pores
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c='blue', marker='o', alpha=0.6, s=5)

        # Labels & settings
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Generated 3D Pore Grid (Density Factor: {density_factor})")
        plt.show()