"""
Uncertainty Estimation for Medical Image Segmentation
Includes Monte Carlo Dropout, Deep Ensembles, and Uncertainty-Guided Refinement
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm


class MonteCarloDropout:
    """Monte Carlo Dropout for uncertainty estimation"""
    
    def __init__(self, model: nn.Module, num_samples: int = 10, dropout_rate: float = 0.1):
        """
        Args:
            model: Model with dropout layers
            num_samples: Number of MC samples
            dropout_rate: Dropout probability
        """
        self.model = model
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate
    
    def enable_dropout(self):
        """Enable dropout during inference"""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
                module.train()
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        return_all_samples: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict with uncertainty estimation
        
        Args:
            x: Input tensor (B, C, H, W)
            return_all_samples: Return all MC samples
            
        Returns:
            Tuple of (mean_prediction, uncertainty_map, all_samples)
        """
        self.model.eval()
        self.enable_dropout()
        
        samples = []
        
        with torch.no_grad():
            for _ in range(self.num_samples):
                output = self.model(x)
                samples.append(output)
        
        # Stack samples
        samples = torch.stack(samples, dim=0)  # (num_samples, B, C, H, W)
        
        # Calculate mean and std
        mean_pred = samples.mean(dim=0)
        uncertainty = samples.std(dim=0)
        
        if return_all_samples:
            return mean_pred, uncertainty, samples
        else:
            return mean_pred, uncertainty, None


class DeepEnsemble:
    """Deep Ensemble for uncertainty estimation"""
    
    def __init__(self, models: List[nn.Module]):
        """
        Args:
            models: List of trained models
        """
        self.models = models
        for model in self.models:
            model.eval()
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with ensemble uncertainty
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Tuple of (mean_prediction, uncertainty_map)
        """
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                output = model(x)
                predictions.append(output)
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)
        
        # Calculate mean and std
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        return mean_pred, uncertainty


class UncertaintyGuidedRefinement:
    """Iterative refinement using uncertainty maps"""
    
    def __init__(
        self,
        model: nn.Module,
        uncertainty_estimator,
        num_iterations: int = 3,
        uncertainty_threshold: float = 0.3
    ):
        """
        Args:
            model: Segmentation model
            uncertainty_estimator: MonteCarloDropout or DeepEnsemble
            num_iterations: Number of refinement iterations
            uncertainty_threshold: Threshold for high uncertainty
        """
        self.model = model
        self.uncertainty_estimator = uncertainty_estimator
        self.num_iterations = num_iterations
        self.uncertainty_threshold = uncertainty_threshold
    
    def refine_prediction(
        self,
        x: torch.Tensor,
        initial_prediction: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Iteratively refine prediction using uncertainty
        
        Args:
            x: Input image (B, C, H, W)
            initial_prediction: Initial prediction (optional)
            
        Returns:
            Tuple of (final_prediction, prediction_history, uncertainty_history)
        """
        prediction_history = []
        uncertainty_history = []
        
        # Get initial prediction
        if initial_prediction is None:
            prediction, uncertainty, _ = self.uncertainty_estimator.predict_with_uncertainty(x)
        else:
            prediction = initial_prediction
            _, uncertainty, _ = self.uncertainty_estimator.predict_with_uncertainty(x)
        
        prediction_history.append(prediction)
        uncertainty_history.append(uncertainty)
        
        for iteration in range(self.num_iterations):
            # Identify high uncertainty regions
            high_uncertainty_mask = uncertainty > self.uncertainty_threshold
            
            if not high_uncertainty_mask.any():
                break
            
            # Create attention mask for refinement
            attention_mask = high_uncertainty_mask.float()
            
            # Refine prediction (simplified - in practice, use prompts or attention)
            with torch.no_grad():
                refined_prediction = self.model(x)
            
            # Blend predictions based on uncertainty
            prediction = torch.where(
                high_uncertainty_mask,
                refined_prediction,
                prediction
            )
            
            # Update uncertainty
            _, uncertainty, _ = self.uncertainty_estimator.predict_with_uncertainty(x)
            
            prediction_history.append(prediction)
            uncertainty_history.append(uncertainty)
        
        return prediction, prediction_history, uncertainty_history


class UncertaintyAnalyzer:
    """Analyze and visualize uncertainty"""
    
    @staticmethod
    def compute_entropy(probabilities: torch.Tensor) -> torch.Tensor:
        """
        Compute predictive entropy
        
        Args:
            probabilities: Predicted probabilities (B, C, H, W)
            
        Returns:
            Entropy map (B, 1, H, W)
        """
        epsilon = 1e-10
        entropy = -torch.sum(
            probabilities * torch.log(probabilities + epsilon),
            dim=1,
            keepdim=True
        )
        return entropy
    
    @staticmethod
    def compute_mutual_information(
        samples: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute mutual information from MC samples
        
        Args:
            samples: MC samples (num_samples, B, C, H, W)
            
        Returns:
            Mutual information map (B, 1, H, W)
        """
        # Mean prediction
        mean_pred = samples.mean(dim=0)
        
        # Entropy of mean
        entropy_mean = UncertaintyAnalyzer.compute_entropy(mean_pred)
        
        # Mean of entropies
        entropies = torch.stack([
            UncertaintyAnalyzer.compute_entropy(sample)
            for sample in samples
        ])
        mean_entropy = entropies.mean(dim=0)
        
        # Mutual information
        mutual_info = entropy_mean - mean_entropy
        
        return mutual_info
    
    @staticmethod
    def get_uncertainty_regions(
        uncertainty_map: torch.Tensor,
        threshold: float = 0.5,
        min_area: int = 100
    ) -> List[np.ndarray]:
        """
        Extract high uncertainty regions
        
        Args:
            uncertainty_map: Uncertainty map (H, W)
            threshold: Uncertainty threshold
            min_area: Minimum region area
            
        Returns:
            List of region coordinates
        """
        import cv2
        
        # Threshold uncertainty map
        binary_map = (uncertainty_map > threshold).cpu().numpy().astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map)
        
        regions = []
        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                region_mask = (labels == i)
                y_coords, x_coords = np.where(region_mask)
                regions.append(np.stack([x_coords, y_coords], axis=1))
        
        return regions


def test_uncertainty_estimation():
    """Test uncertainty estimation"""
    from models.unet import UNetWithDropout
    
    print("Testing Uncertainty Estimation...")
    
    # Create model with dropout
    model = UNetWithDropout(dropout_rate=0.1)
    model.eval()
    
    # Create test input
    x = torch.randn(2, 3, 256, 256)
    
    # Monte Carlo Dropout
    mc_dropout = MonteCarloDropout(model, num_samples=10)
    mean_pred, uncertainty, samples = mc_dropout.predict_with_uncertainty(x, return_all_samples=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Mean prediction shape: {mean_pred.shape}")
    print(f"Uncertainty shape: {uncertainty.shape}")
    print(f"Samples shape: {samples.shape}")
    
    # Compute entropy
    entropy = UncertaintyAnalyzer.compute_entropy(mean_pred)
    print(f"Entropy shape: {entropy.shape}")
    
    # Compute mutual information
    mutual_info = UncertaintyAnalyzer.compute_mutual_information(samples)
    print(f"Mutual information shape: {mutual_info.shape}")
    
    print("\n✅ Uncertainty estimation test passed!")


if __name__ == "__main__":
    test_uncertainty_estimation()
