import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
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

class VAE(pl.LightningModule):
    def __init__(self, latent_dim=128, lr = 0.001, beta=0):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.lr = lr
        self.beta = beta

        # Encoder (3D Convolutional layers)
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1),  # (15x15x15)
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # (8x8x8)
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),  # (4x4x4)
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Latent Space
        self.fc_mu = nn.Linear(128 * 4 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4 * 4, latent_dim)
        
        # Decoder (Deconvolution)
        self.decoder_input = nn.Linear(latent_dim + 1, 128 * 4 * 4 * 4)  # Include density factor
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        x = self.encoder(x)
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, density_factor):
        z = torch.cat([z, density_factor], dim=1)  # Concatenate density factor
        x = self.decoder_input(z)
        x = x.view(-1, 128, 4, 4, 4)
        x = self.decoder(x)
        # x = F.interpolate(x, size=(30, 30, 30), mode='trilinear', align_corners=False) #reshape output matrix
        x = x[:, :, :30, :30, :30]
        return x

    def forward(self, x, density_factor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, density_factor)
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, density, beta, eps=1e-8):
        """
        Custom loss function ensuring exactly `k` ones in the reconstructed grid.

        Parameters:
        - recon_x: Predicted probabilities from the decoder (before thresholding). Shape: (batch_size, 1, 30, 30, 30)
        - x: True binary grid.
        - mu, logvar: Latent space parameters for KL divergence.
        - density: Batch of target densities. Shape: (batch_size,)
        - beta: KL divergence weight.

        Returns:
        - Total loss, reconstruction loss, KL loss.
        """
        batch_size = density.shape[0]  # Get batch size

        # Compute k for each batch element using tensor operations
        k_values = ((density / 0.0882) ** (1 / 0.2632)).long()

        # Flatten the grid for sorting (batch-wise)
        recon_x_flat = recon_x.reshape(batch_size, -1)  # Shape: (batch_size, 30*30*30)
        x_flat = x.reshape(batch_size, -1)  # Shape: (batch_size, 30*30*30)

        # Initialize loss
        total_topk_loss = 0

        # Process each batch sample independently
        for i in range(batch_size):
            k_i = k_values[i].item()  # Extract k for this sample
            if k_i > 0:  # Only apply if at least one '1' is expected
                _, topk_indices = torch.topk(recon_x_flat[i], k_i)  # Top-k highest probabilities
                
                # Create a mask for the k highest values
                mask = torch.zeros_like(recon_x_flat[i])
                mask.scatter_(0, topk_indices, 1)  # Set the k highest indices to 1
                
                # Prevent log(0) errors using clamping
                if mask.sum() > 0:  # Avoid division by zero
                    bce_loss = torch.nn.functional.binary_cross_entropy(
                        torch.clamp(recon_x_flat[i] * mask, eps, 1 - eps),  # Avoid NaN values
                        x_flat[i]*mask,
                        reduction='sum'
                    )
                    total_topk_loss += bce_loss

        # Average loss across batch
        total_topk_loss /= batch_size

        # Prevent KL collapse by using beta warm-up
        kl_weight = min(0.01, self.current_epoch / 100)
        kl_loss = kl_weight * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))

        # Total loss
        loss = total_topk_loss + beta * kl_loss
        return loss, total_topk_loss, kl_loss
    
    # def loss_function(self, recon_x, x, mu, logvar, beta, weight_one=5.0):
    #     """
    #     Custom loss function that emphasizes the reconstruction of ones.
        
    #     Parameters:
    #     - recon_x: Predicted grid (output of decoder).
    #     - x: True grid (ground truth).
    #     - mu, logvar: Latent space variables for KL divergence.
    #     - beta: KL divergence weight (default: 0.001).
    #     - weight_one: Weight factor for ones (default: 5.0).
        
    #     Returns:
    #     - Total loss, reconstruction loss, KL loss.
    #     """
    #     # Compute weighted Binary Cross-Entropy Loss
    #     weight_zero = 1.0  # Keep weight for zeros normal
    #     weights = torch.where(x == 1, weight_one, weight_zero)  # Apply higher weight to ones
    #     recon_loss = nn.functional.binary_cross_entropy(recon_x, x, weight=weights, reduction='sum')

    #     # KL Divergence Loss
    #     kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    #     # Total loss (Reconstruction + KL regularization)
    #     loss = recon_loss + beta * kl_loss
    #     return loss, recon_loss, kl_loss


    def training_step(self, batch, batch_idx):
        x, density_factor = batch
        recon_x, mu, logvar = self.forward(x, density_factor)
        loss, recon_loss, kl_loss = self.loss_function(recon_x, x, mu, logvar, density_factor, self.beta)
        # loss, recon_loss, kl_loss = self.loss_function(recon_x, x, mu, logvar, self.beta)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("reconstruction_loss", recon_loss)
        self.log("kl_divergence", kl_loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, density_factor = batch
        recon_x, mu, logvar = self.forward(x, density_factor)
        loss, recon_loss, kl_loss = self.loss_function(recon_x, x, mu, logvar, density_factor, self.beta)
        # loss, recon_loss, kl_loss = self.loss_function(recon_x, x, mu, logvar, self.beta)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_reconstruction_loss", recon_loss)
        self.log("val_kl_divergence", kl_loss)
        return loss


    def configure_optimizers(self):
        optimizer =  optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
            "gradient_clip_val": 1.0 
        }


