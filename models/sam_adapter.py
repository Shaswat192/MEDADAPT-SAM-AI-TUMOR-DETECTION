"""
SAM Adapter Model for Medical Image Segmentation
Lightweight adapter modules for domain adaptation
"""

import torch
import torch.nn as nn
from segment_anything import sam_model_registry, SamPredictor
from typing import Optional, Tuple, List
import numpy as np


class Adapter(nn.Module):
    """Bottleneck Adapter Module"""
    
    def __init__(self, dim: int, adapter_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.down_proj = nn.Linear(dim, adapter_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(adapter_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        return x + residual


class SAMAdapter(nn.Module):
    """
    SAM with Adapter Modules for Medical Image Segmentation
    
    Args:
        model_type: SAM model type ('vit_b', 'vit_l', 'vit_h')
        checkpoint_path: Path to SAM checkpoint
        adapter_dim: Dimension of adapter bottleneck
        num_adapters: Number of adapter layers
        freeze_sam: Freeze original SAM weights
    """
    
    def __init__(
        self,
        model_type: str = 'vit_b',
        checkpoint_path: Optional[str] = None,
        adapter_dim: int = 64,
        num_adapters: int = 12,
        freeze_sam: bool = True,
        num_classes: int = 3
    ):
        super().__init__()
        
        # Load SAM model
        if checkpoint_path:
            self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        else:
            self.sam = sam_model_registry[model_type]()
        
        # Freeze SAM if specified
        if freeze_sam:
            for param in self.sam.parameters():
                param.requires_grad = False
        
        # Get embedding dimension
        if model_type == 'vit_b':
            embed_dim = 768
        elif model_type == 'vit_l':
            embed_dim = 1024
        else:  # vit_h
            embed_dim = 1280
        
        # Add adapters to image encoder
        self.adapters = nn.ModuleList([
            Adapter(embed_dim, adapter_dim) 
            for _ in range(num_adapters)
        ])
        
        # Multi-class output head
        self.num_classes = num_classes
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        
    def forward(
        self,
        images: torch.Tensor,
        prompts: Optional[dict] = None,
        multimask_output: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with optional prompts
        
        Args:
            images: Input images (B, 3, H, W)
            prompts: Dictionary containing 'boxes' and/or 'points'
            multimask_output: Return multiple masks
            
        Returns:
            Segmentation masks (B, num_classes, H, W)
        """
        batch_size = images.shape[0]
        
        # Get image embeddings
        image_embeddings = self.sam.image_encoder(images)
        
        # Apply adapters (simplified - in practice, inject into transformer blocks)
        # This is a placeholder for adapter integration
        
        if prompts is not None:
            # Use SAM's prompt encoder
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=prompts.get('points'),
                boxes=prompts.get('boxes'),
                masks=None
            )
            
            # Decode with prompts
            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output
            )
        else:
            # No prompts - use learned embeddings
            low_res_masks = image_embeddings
        
        # Upscale to full resolution
        masks = self.output_upscaling(low_res_masks)
        
        return torch.sigmoid(masks)
    
    def freeze_adapters(self):
        """Freeze adapter weights"""
        for adapter in self.adapters:
            for param in adapter.parameters():
                param.requires_grad = False
    
    def unfreeze_adapters(self):
        """Unfreeze adapter weights"""
        for adapter in self.adapters:
            for param in adapter.parameters():
                param.requires_grad = True
    
    def get_num_trainable_params(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SAMWithAutomaticPrompts(nn.Module):
    """SAM Adapter with Automatic Prompt Generation"""
    
    def __init__(self, sam_adapter: SAMAdapter):
        super().__init__()
        self.sam_adapter = sam_adapter
        
        # Prompt generation network
        self.prompt_generator = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # Generate box coordinates
        )
    
    def generate_box_prompts(self, images: torch.Tensor) -> torch.Tensor:
        """
        Generate box prompts from images
        
        Args:
            images: Input images (B, 3, H, W)
            
        Returns:
            Box coordinates (B, 4) in format [x1, y1, x2, y2]
        """
        boxes = self.prompt_generator(images)
        
        # Normalize to image dimensions
        boxes = torch.sigmoid(boxes)  # 0-1 range
        
        H, W = images.shape[2:]
        boxes[:, [0, 2]] *= W  # x coordinates
        boxes[:, [1, 3]] *= H  # y coordinates
        
        return boxes
    
    def forward(self, images: torch.Tensor, use_auto_prompts: bool = True) -> torch.Tensor:
        """
        Forward pass with automatic prompt generation
        
        Args:
            images: Input images (B, 3, H, W)
            use_auto_prompts: Use automatically generated prompts
            
        Returns:
            Segmentation masks (B, num_classes, H, W)
        """
        if use_auto_prompts:
            boxes = self.generate_box_prompts(images)
            prompts = {'boxes': boxes, 'points': None}
        else:
            prompts = None
        
        return self.sam_adapter(images, prompts)


def load_sam_adapter(
    model_type: str = 'vit_b',
    checkpoint_path: Optional[str] = None,
    adapter_dim: int = 64,
    freeze_sam: bool = True
) -> SAMAdapter:
    """
    Load SAM Adapter model
    
    Args:
        model_type: SAM model type
        checkpoint_path: Path to SAM checkpoint
        adapter_dim: Adapter dimension
        freeze_sam: Freeze SAM weights
        
    Returns:
        SAMAdapter model
    """
    model = SAMAdapter(
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        adapter_dim=adapter_dim,
        freeze_sam=freeze_sam
    )
    
    print(f"Loaded SAM Adapter ({model_type})")
    print(f"Trainable parameters: {model.get_num_trainable_params():,}")
    
    return model


if __name__ == "__main__":
    # Test SAM Adapter
    print("Testing SAM Adapter...")
    
    # Note: Requires SAM checkpoint to be downloaded
    model = SAMAdapter(
        model_type='vit_b',
        checkpoint_path=None,  # Set path to SAM checkpoint
        adapter_dim=64,
        freeze_sam=True
    )
    
    # Test forward pass
    images = torch.randn(2, 3, 256, 256)
    output = model(images)
    
    print(f"Input shape: {images.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Trainable params: {model.get_num_trainable_params():,}")
