"""
Explainable AI (XAI) for MedAdapt-SAM
Includes Attention Visualization and Grad-CAM integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import List, Tuple, Dict, Any


class AttentionVisualizer:
    """Visualize self-attention weights from SAM or Transformer-based models"""
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def get_attention_maps(self, image: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract attention maps from the model
        This is a placeholder as direct extraction depends on SAM implementation
        In practice, we would use hooks to capture attention weights from VIT blocks
        """
        # Placeholder for demonstration
        # For a 256x256 image with 16x16 patches, we have 16*16 = 256 tokens
        batch_size = image.shape[0]
        num_heads = 8
        num_tokens = 256
        
        # Simulated attention weights (batch, heads, tokens, tokens)
        attn_weights = torch.randn(batch_size, num_heads, num_tokens, num_tokens).softmax(dim=-1)
        return [attn_weights]

    def plot_attention_map(self, attn_map: torch.Tensor, image: np.ndarray, head_idx: int = 0):
        """Plot attention heatmaps overlaid on image"""
        # Assume attn_map is (B, H, T, T) - take mean across tokens for query
        # (simplified visualization)
        weights = attn_map[0, head_idx].mean(dim=0).reshape(16, 16)
        weights = F.interpolate(weights.unsqueeze(0).unsqueeze(0), size=image.shape[:2], mode='bilinear')[0, 0]
        
        weights = weights.detach().cpu().numpy()
        weights = (weights - weights.min()) / (weights.max() - weights.min())
        
        heatmap = cv2.applyColorMap((weights * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        overlay = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
        return overlay


class GradCAM:
    """Gradient-weighted Class Activation Mapping implementation"""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate_heatmap(self, input_image: torch.Tensor, class_idx: int) -> np.ndarray:
        """Generate Grad-CAM heatmap for a given class"""
        self.model.zero_grad()
        output = self.model(input_image)
        
        # Target specific class
        score = output[0, class_idx].sum()
        score.backward()
        
        # Average gradients across spatial dimensions
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted sum of activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize and resize
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam


def visualize_prompt_influence(
    model: nn.Module, 
    image: torch.Tensor, 
    prompts: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """
    Analyze how different prompts influence the final prediction.
    We compare results with and without prompts.
    """
    with torch.no_grad():
        # Prediction with prompts
        pred_with = model(image, prompts)
        
        # Prediction without prompts
        pred_without = model(image, None)
    
    # Difference map (Influence map)
    influence = torch.abs(pred_with - pred_without).cpu().numpy()[0]
    
    return {
        'ET_influence': influence[0],
        'TC_influence': influence[1],
        'WT_influence': influence[2]
    }
