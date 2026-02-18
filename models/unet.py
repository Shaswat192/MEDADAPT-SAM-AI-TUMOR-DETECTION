"""
U-Net Model for Brain Tumor Segmentation (Baseline)
Classic U-Net architecture with skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DoubleConv(nn.Module):
    """Double Convolution Block"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output Convolution"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net Architecture for Medical Image Segmentation
    
    Args:
        in_channels: Number of input channels (3 for RGB)
        out_channels: Number of output channels (3 for ET, TC, WT)
        features: List of feature dimensions for each level
        bilinear: Use bilinear upsampling instead of transposed conv
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        features: List[int] = [64, 128, 256, 512],
        bilinear: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # Initial convolution
        self.inc = DoubleConv(in_channels, features[0])
        
        # Encoder (downsampling)
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        
        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down(features[3], features[3] * 2 // factor)
        
        # Decoder (upsampling)
        self.up1 = Up(features[3] * 2, features[3] // factor, bilinear)
        self.up2 = Up(features[3], features[2] // factor, bilinear)
        self.up3 = Up(features[2], features[1] // factor, bilinear)
        self.up4 = Up(features[1], features[0], bilinear)
        
        # Output
        self.outc = OutConv(features[0], out_channels)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return torch.sigmoid(logits)
    
    def get_num_params(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UNetWithDropout(UNet):
    """U-Net with Monte Carlo Dropout for uncertainty estimation"""
    
    def __init__(self, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        
        # Add dropout layers
        self.dropout = nn.Dropout2d(p=dropout_rate)
    
    def forward(self, x, apply_dropout: bool = False):
        # Encoder
        x1 = self.inc(x)
        if apply_dropout:
            x1 = self.dropout(x1)
            
        x2 = self.down1(x1)
        if apply_dropout:
            x2 = self.dropout(x2)
            
        x3 = self.down2(x2)
        if apply_dropout:
            x3 = self.dropout(x3)
            
        x4 = self.down3(x3)
        if apply_dropout:
            x4 = self.dropout(x4)
            
        x5 = self.down4(x4)
        if apply_dropout:
            x5 = self.dropout(x5)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return torch.sigmoid(logits)


def test_unet():
    """Test U-Net model"""
    model = UNet(in_channels=3, out_channels=3)
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {model.get_num_params():,}")
    
    # Test with dropout
    model_dropout = UNetWithDropout(dropout_rate=0.1)
    output_dropout = model_dropout(x, apply_dropout=True)
    print(f"Output with dropout shape: {output_dropout.shape}")


if __name__ == "__main__":
    test_unet()
