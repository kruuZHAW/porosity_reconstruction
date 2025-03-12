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

class VAE(pl.LightningModule):
    def __init__(self, latent_dim=128, lr = 0.001, beta=1, alpha=0, gamma=0):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.lr = lr
        self.beta = beta # KL regularization
        self.alpha = alpha # Number of ones regularization
        self.gamma = gamma # L1 Regularisation for sparsity

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
            nn.Sigmoid()  # Ensures output is in range (0,1) for occupancy prediction
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
        # Reconstruction loss
        criterion = nn.BCELoss(reduction='mean') 
        # criterion = nn.MSELoss(reduction = 'mean')
        reco_loss = criterion(recon_grid, grid)

        # Number of ones loss (MSE instead of squared error difference)
        ones_diff = ((recon_grid.mean() - grid.mean()) ** 2) * self.alpha 
        ones_diff /= mu.shape[0] # normalise

        # L1 Sparsity Regularization (penalizes large activations)
        l1_reg = torch.norm(recon_grid, p=1) * self.gamma 
        l1_reg /= mu.shape[0]

        # KL Divergence Loss
        # kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * beta
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=1),dim=0)

        # Total Loss
        loss = reco_loss + kl_loss + ones_diff + l1_reg
        return loss, reco_loss, kl_loss



    def training_step(self, batch, batch_idx):
        x, density_factor = batch
        recon_x, mu, logvar = self.forward(x, density_factor)

        # Gradually increase beta (KL loss weight)
        beta = min(self.beta, self.current_epoch / 10.0)
        loss, recon_loss, kl_loss = self.loss_function(recon_x, x, mu, logvar, beta)
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("reconstruction_loss", recon_loss)
        self.log("kl_divergence", kl_loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, density_factor = batch
        recon_x, mu, logvar = self.forward(x, density_factor)
        
        # Gradually increase beta (KL loss weight)
        beta = min(self.beta, self.current_epoch / 10.0)
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


