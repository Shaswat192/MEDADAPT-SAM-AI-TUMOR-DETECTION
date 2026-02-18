"""
Training Script for U-Net Baseline Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
from tqdm import tqdm
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.unet import UNet, UNetWithDropout
from data.dataset_loader import create_dataloaders
from evaluation.metrics import MetricsCalculator, dice_coefficient
from torch.utils.tensorboard import SummaryWriter


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.contiguous()
        target = target.contiguous()
        
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Combined Dice + Focal + Boundary Loss"""
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.3,
        boundary_weight: float = 0.2
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        
        # Simplified boundary loss (can be enhanced)
        boundary = nn.functional.binary_cross_entropy(pred, target)
        
        total_loss = (
            self.dice_weight * dice +
            self.focal_weight * focal +
            self.boundary_weight * boundary
        )
        
        return total_loss


class Trainer:
    """Trainer for U-Net model"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss function
        self.criterion = CombinedLoss(
            dice_weight=config['training']['loss']['dice_weight'],
            focal_weight=config['training']['loss']['focal_weight'],
            boundary_weight=config['training']['loss']['boundary_weight']
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['num_epochs']
        )
        
        # Metrics
        self.metrics_calculator = MetricsCalculator(
            num_classes=3,
            class_names=['ET', 'TC', 'WT']
        )
        
        # Logging
        self.writer = SummaryWriter(log_dir='logs/unet')
        
        # Best model tracking
        self.best_dice = 0.0
        self.patience_counter = 0
        self.patience = config['training']['early_stopping_patience']
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate loss
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to tensorboard
            if batch_idx % self.config['logging']['log_interval'] == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), step)
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self, epoch: int) -> dict:
        """Validate model"""
        self.model.eval()
        self.metrics_calculator.reset()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                
                # Update metrics
                self.metrics_calculator.update(outputs, masks)
        
        # Compute metrics
        metrics = self.metrics_calculator.compute()
        avg_loss = total_loss / len(self.val_loader)
        
        # Log to tensorboard
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/Dice', metrics['dice_mean'], epoch)
        self.writer.add_scalar('Val/IoU', metrics['iou_mean'], epoch)
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }
        
        # Save regular checkpoint
        checkpoint_path = f"checkpoints/unet_epoch_{epoch}.pth"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = "checkpoints/unet_best.pth"
            torch.save(self.model.state_dict(), best_path)
            print(f"✅ Saved best model with Dice: {metrics['dice_mean']:.4f}")
    
    def train(self):
        """Full training loop"""
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Number of epochs: {self.config['training']['num_epochs']}")
        
        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.config['training']['num_epochs']}")
            print(f"{'='*60}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            metrics = self.validate(epoch)
            print(f"\nValidation Metrics:")
            print(f"  Dice: {metrics['dice_mean']:.4f}")
            print(f"  IoU:  {metrics['iou_mean']:.4f}")
            print(f"  HD95: {metrics['hd95_mean']:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Save checkpoint
            is_best = metrics['dice_mean'] > self.best_dice
            if is_best:
                self.best_dice = metrics['dice_mean']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if epoch % self.config['logging']['save_checkpoint_interval'] == 0:
                self.save_checkpoint(epoch, metrics, is_best)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\n⚠️ Early stopping triggered after {epoch} epochs")
                break
        
        print(f"\n✅ Training complete! Best Dice: {self.best_dice:.4f}")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train U-Net for brain tumor segmentation')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Loading dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=config['dataset']['data_path'],
        batch_size=config['training']['batch_size'],
        image_size=config['dataset']['image_size'],
        num_workers=4,
        train_split=config['dataset']['train_split'],
        val_split=config['dataset']['val_split'],
        test_split=config['dataset']['test_split']
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = UNet(
        in_channels=config['model']['unet']['in_channels'],
        out_channels=config['model']['unet']['out_channels'],
        features=config['model']['unet']['features']
    )
    
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
