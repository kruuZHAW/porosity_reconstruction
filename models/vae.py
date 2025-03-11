import torch
import torch.nn as nn
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

        # Convert to 3D 30x30x30 matrix
        grid = np.zeros((30, 30, 30), dtype=np.float32)
        for row in data:
            x, y, z, occ = row
            grid[int(x), int(y), int(z)] = occ
        
        # Extract density from filename
        density_factor = float(file_path.split("_")[-1].replace(".npy", ""))
        
        # Convert to tensors
        grid_tensor = torch.tensor(grid).unsqueeze(0)  # Shape (1,30,30,30)
        density_tensor = torch.tensor([density_factor], dtype=torch.float32)

        return grid_tensor, density_tensor

class VAE(pl.LightningModule):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

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
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (8x8x8)
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (15x15x15)
            nn.ReLU(),
            nn.ConvTranspose3d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (30x30x30)
            nn.Sigmoid()  # Output probability map
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
        return x

    def forward(self, x, density_factor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, density_factor)
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        # Reconstruction loss (Binary Cross-Entropy)
        recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        # KL Divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss

    def training_step(self, batch, batch_idx):
        x, density_factor = batch
        recon_x, mu, logvar = self.forward(x, density_factor)
        loss = self.loss_function(recon_x, x, mu, logvar)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


