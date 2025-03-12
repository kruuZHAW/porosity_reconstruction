import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np

import os
import sys
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from utils.poreDataset import PoreDataset

# ----------------------------
# CNN Model (Density → Grid)
# ----------------------------
class CNNGenerator(pl.LightningModule):
    def __init__(self, lr=0.001, alpha = 1, beta = 1):
        super(CNNGenerator, self).__init__()
        self.lr = lr
        self.alpha = alpha # regulariation for number of ones in grid
        self.beta = beta # l1 regularisation

        # Fully Connected to Expand Density
        self.fc = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * 4 * 4 * 4),
            nn.ReLU(),
        )


        # Upsampling + Conv3D Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),  # (4 → 8)
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),  # (8 → 16)
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.Upsample(scale_factor=(1.875, 1.875, 1.875), mode='trilinear', align_corners=True),  # (16 → 30)
            nn.Conv3d(32, 1, kernel_size=3, padding=1),
            # nn.Sigmoid()
        )
        
        def init_weights(m):
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.xavier_uniform_(m.weight)
        self.apply(init_weights)

    def forward(self, density_factor):
        """
        Given a density factor, generate a 3D grid.
        """
        x = self.fc(density_factor)  # Expand density factor
        x = x.view(-1, 128, 4, 4, 4)  # Reshape to match decoder input
        x = self.decoder(x)  # Decode to 3D grid
        return x

    def training_step(self, batch, batch_idx):
        """
        Training step: Forward pass + loss computation.
        """
        grid, density_factor = batch
        recon_grid = self.forward(density_factor)
        # loss = self.loss_function(recon_grid, grid, density_factor)
        criterion = nn.BCEWithLogitsLoss(reduction='sum')
        loss = criterion(recon_grid, grid)
        loss += self.alpha*(recon_grid.sum() - grid.sum())**2 # ensure to have the same number of ones
        loss += self.beta*(recon_grid.sum()) # force sparsity

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step: Compute loss.
        """
        grid, density_factor = batch 
        recon_grid = self.forward(density_factor)
        # loss = self.loss_function(recon_grid, grid, density_factor)
        criterion = nn.BCEWithLogitsLoss(reduction='sum')
        loss = criterion(recon_grid, grid)
        loss += self.alpha*(recon_grid.sum() - grid.sum())**2 # ensure to have the same number of ones
        loss += self.beta*(recon_grid.sum()) # force sparsity
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """
        Define optimizer and learning rate scheduler.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

        return {
            "optimizer": optimizer, 
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
            "gradient_clip_val": 1.0 
        }
