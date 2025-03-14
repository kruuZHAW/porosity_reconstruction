import os
import sys
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from utils.poreDataset import PoreDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalBCELossWithConstraints(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, size=30, radius=2, lambda_edge=1.0, lambda_cluster=1.0, device='cpu'):
        """
        Modified Focal Binary Cross-Entropy Loss with:
        - Soft edge penalty: discourages ones near grid boundaries.
        - Clustering penalty: encourages ones to form spherical clusters.

        Args:
            alpha (float): Class balancing factor for focal loss (default 0.25).
            gamma (float): Focusing parameter for focal loss (default 2.0).
            size (int): Size of the cubic grid (default 30x30x30).
            radius (int): Radius of the spherical kernel for clustering constraint.
            lambda_edge (float): Weight of edge penalty term (default 1.0).
            lambda_cluster (float): Weight of clustering penalty term (default 1.0).
            device (str): Device where the tensors should be stored ('cpu' or 'cuda').
        """
        super(FocalBCELossWithConstraints, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_edge = lambda_edge
        self.lambda_cluster = lambda_cluster
        self.device = device

        # Compute edge mask and spherical kernel internally
        self.edge_mask = self._compute_edge_mask(size).to(device)
        self.spherical_kernel = self._compute_spherical_kernel(radius).to(device)

    def _compute_edge_mask(self, size):
        """Creates an edge mask where values are high near edges and low in the center."""
        coords = torch.stack(torch.meshgrid(
            torch.arange(size, dtype=torch.float32),
            torch.arange(size, dtype=torch.float32),
            torch.arange(size, dtype=torch.float32),
            indexing="ij"
        ), dim=0)  # Shape: [3, size, size, size]

        dist_x = torch.minimum(coords[0], size - 1 - coords[0])
        dist_y = torch.minimum(coords[1], size - 1 - coords[1])
        dist_z = torch.minimum(coords[2], size - 1 - coords[2])

        dist_to_edge = torch.minimum(dist_x, torch.minimum(dist_y, dist_z))
        edge_mask = 1.0 - (dist_to_edge / dist_to_edge.max())  # Normalize and invert
        return edge_mask.unsqueeze(0).unsqueeze(0)  # Shape [1,1,size,size,size]

    def _compute_spherical_kernel(self, radius):
        """Creates a 3D spherical kernel for convolution-based clustering penalty."""
        diam = 2 * radius + 1  # Kernel size
        grid_range = torch.arange(-radius, radius + 1, dtype=torch.float32)

        Z, Y, X = torch.meshgrid(grid_range, grid_range, grid_range, indexing="ij")
        sphere = (X**2 + Y**2 + Z**2) <= (radius**2)  # 1 inside sphere, 0 outside

        return sphere.float().unsqueeze(0).unsqueeze(0)  # Shape [1,1,D,D,D]

    def forward(self, preds, targets):
        """
        Compute the loss.

        Args:
            preds (torch.Tensor): Predicted probability map after sigmoid (shape [B,1,30,30,30]).
            targets (torch.Tensor): Ground truth binary labels (shape [B,1,30,30,30]).

        Returns:
            torch.Tensor: Total loss (focal BCE + edge penalty + clustering penalty).
        """
        # --- Standard BCE Focal Loss ---
        bce_loss = F.binary_cross_entropy(preds, targets, reduction='none')
        pt = targets * preds + (1 - targets) * (1 - preds)  # p_t for each voxel
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = (1.0 - pt) ** self.gamma
        focal_loss = alpha_factor * focal_weight * bce_loss
        focal_loss = focal_loss.mean()  # Reduce to scalar

        # --- Soft Edge Penalty ---
        if self.lambda_edge > 0:
            edge_penalty = (preds * self.edge_mask).mean()  # High if ones appear near edges
        else:
            edge_penalty = 0

        # --- Clustering Penalty ---
        if self.lambda_cluster > 0:
            density_map = F.conv3d(preds, self.spherical_kernel, padding=self.spherical_kernel.shape[-1]//2)
            density_map = density_map / (self.spherical_kernel.sum() + 1e-8)  # Normalize density to [0,1]
            cluster_penalty = (preds * (1.0 - density_map)).mean()  # Penalize isolated ones
        else:
            cluster_penalty = 0
            
        # --- Total Loss ---
        total_loss = focal_loss + self.lambda_edge * edge_penalty + self.lambda_cluster * cluster_penalty
        return total_loss
    

class VAE(pl.LightningModule):
    def __init__(self, latent_dim, lr, alpha, gamma, beta=1, lambda_edge = 1.0, lambda_cluster=1.0):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.lr = lr
        self.beta = beta # KL regularization
        self.alpha = alpha # Class Balancing factor Focal Loss
        self.gamma = gamma # Focusing parameter for focal loss
        self.lambda_edge = lambda_edge # Regularisation for soft edge constraint
        self.lambda_cluster = lambda_cluster # Regularisation for spherical aggregation of ones

        # Encoder (3D Convolutional layers)
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1),  # (15x15x15)
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),  # Keep (15x15x15)
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),  # (8x8x8)
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),  # (4x4x4)
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Latent Space
        self.fc_mu = nn.Linear(256 * 4 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4 * 4, latent_dim)

        # Decoder Input Layer
        self.decoder_input = nn.Linear(latent_dim + 1, 256 * 4 * 4 * 4)  # Includes density factor
        # self.decoder_input = nn.Linear(latent_dim, 256 * 4 * 4 * 4)

        # Upsampling + Conv3D Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),  # (4 → 8)
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),  # (8 → 16)
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.Upsample(scale_factor=(15/8, 15/8, 15/8), mode='trilinear', align_corners=True),  # (16 → 15)
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.Upsample(scale_factor=1, mode='trilinear', align_corners=True),  # (15 → 30)
            nn.Conv3d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        
        # Xavier initialization for weights
        def weights_init(m):
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(weights_init)
    
    def encode(self, x):
        x = self.encoder(x)
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # Sample from a standard NN
        return mu + eps * std

    def decode(self, z, density_factor):
        z = torch.cat([z, density_factor], dim=1)  # Concatenate density factor
        x = self.decoder_input(z)
        x = x.view(-1, 256, 4, 4, 4)
        x = self.decoder(x)
        return x

    def forward(self, x, density_factor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, density_factor)
        return recon_x, mu, logvar
    

    
    def loss_function(self, recon_grid, grid, mu, logvar, beta):
            # Use Free-Bits Regularization: Prevent KL from collapsing by setting a minimum threshold for KLD
        def kl_loss(mu, logvar, beta, min_kl=0.1):
            kl = -0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(dim=1).mean()
            return beta * torch.max(kl, torch.tensor(min_kl, device=mu.device)) 

        # criterion = nn.BCELoss(reduction='mean') 
        criterion = FocalBCELossWithConstraints(
            size=30, 
            radius=2, 
            alpha=self.alpha, 
            gamma=self.gamma, 
            lambda_edge=self.lambda_edge, 
            lambda_cluster=self.lambda_cluster, 
            device='cpu'
    )
        reco_loss = criterion(recon_grid, grid)

        # Number of ones reg
        # ones_diff = ((recon_grid.mean() - grid.mean()) ** 2) * self.alpha 

        # L1 Sparsity Regularization (penalizes large activations)
        # We try to match the sparisty of the recontruction with the sparisty of the ground truth
        # l1_reg = ((torch.norm(recon_grid, p=1) - torch.norm(grid, p=1))  * self.gamma) / (27000 * mu.shape[0]) #normalize by grid size and batch size

        # Free Bits KL Divergence
        kl = kl_loss(mu, logvar, beta)

        loss = reco_loss + beta * kl
        return loss, reco_loss, kl


    def training_step(self, batch, batch_idx):
        x, density_factor = batch
        recon_x, mu, logvar = self.forward(x, density_factor)
        
        #Skip KL for first few epochs
        if self.current_epoch < 5:
            beta = 0
        else:
            # Cyclical annealing for KL loss
            cycle_length = 20
            beta = self.beta * (1 - np.cos(np.pi * (self.current_epoch % cycle_length) / cycle_length)) / 2

        loss, recon_loss, kl_loss = self.loss_function(recon_x, x, mu, logvar, beta)
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("reconstruction_loss", recon_loss)
        self.log("kl_divergence", kl_loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, density_factor = batch
        recon_x, mu, logvar = self.forward(x, density_factor)
        
        #Skip KL for first few epochs
        if self.current_epoch < 5:
            beta = 0
        else:
            # Cyclical annealing for KL loss
            cycle_length = 20
            beta = self.beta * (1 - np.cos(np.pi * (self.current_epoch % cycle_length) / cycle_length)) / 2

        loss, recon_loss, kl_loss = self.loss_function(recon_x, x, mu, logvar, beta)
        
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

