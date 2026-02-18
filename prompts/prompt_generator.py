"""
Automatic Prompt Generation for SAM
Includes box, point, and hybrid prompt strategies
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Optional
import cv2
from scipy.ndimage import center_of_mass, label


class PromptGenerator:
    """Generate prompts for SAM from masks or predictions"""
    
    def __init__(self, image_size: int = 256):
        self.image_size = image_size
    
    def generate_box_from_mask(
        self,
        mask: np.ndarray,
        margin: int = 5
    ) -> Optional[np.ndarray]:
        """
        Generate bounding box from binary mask
        
        Args:
            mask: Binary mask (H, W)
            margin: Margin to add around box
            
        Returns:
            Box coordinates [x1, y1, x2, y2] or None if empty mask
        """
        if mask.sum() == 0:
            return None
        
        # Find non-zero pixels
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        
        # Add margin
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(mask.shape[1] - 1, x2 + margin)
        y2 = min(mask.shape[0] - 1, y2 + margin)
        
        return np.array([x1, y1, x2, y2])
    
    def generate_points_from_mask(
        self,
        mask: np.ndarray,
        num_positive: int = 5,
        num_negative: int = 3,
        strategy: str = 'random'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate point prompts from mask
        
        Args:
            mask: Binary mask (H, W)
            num_positive: Number of positive points
            num_negative: Number of negative points
            strategy: 'random', 'centroid', 'boundary'
            
        Returns:
            Tuple of (positive_points, negative_points) each (N, 2) [x, y]
        """
        positive_points = []
        negative_points = []
        
        if mask.sum() == 0:
            return np.array([]), np.array([])
        
        # Generate positive points
        if strategy == 'random':
            # Random sampling from mask
            y_coords, x_coords = np.where(mask > 0)
            if len(y_coords) > 0:
                indices = np.random.choice(len(y_coords), min(num_positive, len(y_coords)), replace=False)
                positive_points = np.stack([x_coords[indices], y_coords[indices]], axis=1)
        
        elif strategy == 'centroid':
            # Use centroid and nearby points
            cy, cx = center_of_mass(mask)
            positive_points = [[int(cx), int(cy)]]
            
            # Add nearby points
            y_coords, x_coords = np.where(mask > 0)
            if len(y_coords) > 0:
                distances = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
                sorted_indices = np.argsort(distances)
                for i in range(1, min(num_positive, len(sorted_indices))):
                    idx = sorted_indices[i * len(sorted_indices) // num_positive]
                    positive_points.append([x_coords[idx], y_coords[idx]])
            
            positive_points = np.array(positive_points)
        
        elif strategy == 'boundary':
            # Sample from boundary
            boundary = cv2.Canny((mask * 255).astype(np.uint8), 100, 200)
            y_coords, x_coords = np.where(boundary > 0)
            if len(y_coords) > 0:
                indices = np.random.choice(len(y_coords), min(num_positive, len(y_coords)), replace=False)
                positive_points = np.stack([x_coords[indices], y_coords[indices]], axis=1)
        
        # Generate negative points (outside mask)
        if num_negative > 0:
            y_coords, x_coords = np.where(mask == 0)
            if len(y_coords) > 0:
                indices = np.random.choice(len(y_coords), min(num_negative, len(y_coords)), replace=False)
                negative_points = np.stack([x_coords[indices], y_coords[indices]], axis=1)
        
        return positive_points, negative_points
    
    def generate_prompts_from_uncertainty(
        self,
        uncertainty_map: np.ndarray,
        num_points: int = 5,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Generate corrective point prompts from uncertainty map
        
        Args:
            uncertainty_map: Uncertainty map (H, W)
            num_points: Number of points to generate
            threshold: Uncertainty threshold
            
        Returns:
            Point coordinates (N, 2) [x, y]
        """
        # Find high uncertainty regions
        high_uncertainty = uncertainty_map > threshold
        
        if high_uncertainty.sum() == 0:
            return np.array([])
        
        # Sample points from high uncertainty regions
        y_coords, x_coords = np.where(high_uncertainty)
        
        # Weight by uncertainty value
        weights = uncertainty_map[y_coords, x_coords]
        weights = weights / weights.sum()
        
        indices = np.random.choice(
            len(y_coords),
            min(num_points, len(y_coords)),
            replace=False,
            p=weights
        )
        
        points = np.stack([x_coords[indices], y_coords[indices]], axis=1)
        return points


class BatchPromptGenerator:
    """Generate prompts for batches of images"""
    
    def __init__(self, image_size: int = 256):
        self.generator = PromptGenerator(image_size)
    
    def generate_box_prompts(
        self,
        masks: torch.Tensor,
        margin: int = 5
    ) -> torch.Tensor:
        """
        Generate box prompts for batch
        
        Args:
            masks: Batch of masks (B, C, H, W)
            margin: Box margin
            
        Returns:
            Box coordinates (B, C, 4)
        """
        batch_size, num_classes = masks.shape[:2]
        boxes = torch.zeros(batch_size, num_classes, 4)
        
        masks_np = masks.cpu().numpy()
        
        for b in range(batch_size):
            for c in range(num_classes):
                box = self.generator.generate_box_from_mask(
                    masks_np[b, c],
                    margin=margin
                )
                if box is not None:
                    boxes[b, c] = torch.from_numpy(box)
        
        return boxes
    
    def generate_point_prompts(
        self,
        masks: torch.Tensor,
        num_positive: int = 5,
        num_negative: int = 3,
        strategy: str = 'random'
    ) -> Dict[str, List]:
        """
        Generate point prompts for batch
        
        Args:
            masks: Batch of masks (B, C, H, W)
            num_positive: Number of positive points per class
            num_negative: Number of negative points per class
            strategy: Sampling strategy
            
        Returns:
            Dictionary with 'positive' and 'negative' point lists
        """
        batch_size, num_classes = masks.shape[:2]
        masks_np = masks.cpu().numpy()
        
        positive_prompts = []
        negative_prompts = []
        
        for b in range(batch_size):
            batch_positive = []
            batch_negative = []
            
            for c in range(num_classes):
                pos, neg = self.generator.generate_points_from_mask(
                    masks_np[b, c],
                    num_positive=num_positive,
                    num_negative=num_negative,
                    strategy=strategy
                )
                batch_positive.append(pos)
                batch_negative.append(neg)
            
            positive_prompts.append(batch_positive)
            negative_prompts.append(batch_negative)
        
        return {
            'positive': positive_prompts,
            'negative': negative_prompts
        }
    
    def generate_hybrid_prompts(
        self,
        masks: torch.Tensor,
        box_margin: int = 5,
        num_positive: int = 3,
        num_negative: int = 2
    ) -> Dict[str, torch.Tensor]:
        """
        Generate hybrid prompts (boxes + points)
        
        Args:
            masks: Batch of masks (B, C, H, W)
            box_margin: Box margin
            num_positive: Number of positive points
            num_negative: Number of negative points
            
        Returns:
            Dictionary with 'boxes' and 'points'
        """
        boxes = self.generate_box_prompts(masks, margin=box_margin)
        points = self.generate_point_prompts(
            masks,
            num_positive=num_positive,
            num_negative=num_negative
        )
        
        return {
            'boxes': boxes,
            'points': points
        }


class UncertaintyGuidedPromptGenerator:
    """Generate prompts guided by uncertainty maps"""
    
    def __init__(self, image_size: int = 256):
        self.generator = PromptGenerator(image_size)
    
    def generate_corrective_prompts(
        self,
        prediction: np.ndarray,
        uncertainty_map: np.ndarray,
        num_points: int = 5,
        uncertainty_threshold: float = 0.5
    ) -> np.ndarray:
        """
        Generate corrective prompts from uncertainty
        
        Args:
            prediction: Current prediction (H, W)
            uncertainty_map: Uncertainty map (H, W)
            num_points: Number of corrective points
            uncertainty_threshold: Uncertainty threshold
            
        Returns:
            Corrective point coordinates (N, 2)
        """
        return self.generator.generate_prompts_from_uncertainty(
            uncertainty_map,
            num_points=num_points,
            threshold=uncertainty_threshold
        )
    
    def iterative_refinement(
        self,
        model,
        image: torch.Tensor,
        initial_prompts: Dict,
        num_iterations: int = 3,
        uncertainty_threshold: float = 0.5
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Iteratively refine segmentation using uncertainty
        
        Args:
            model: Segmentation model
            image: Input image
            initial_prompts: Initial prompts
            num_iterations: Number of refinement iterations
            uncertainty_threshold: Uncertainty threshold
            
        Returns:
            Tuple of (final_prediction, prompt_history)
        """
        current_prompts = initial_prompts
        prompt_history = [initial_prompts]
        
        for iteration in range(num_iterations):
            # Get prediction
            with torch.no_grad():
                prediction = model(image, current_prompts)
            
            # Estimate uncertainty (simplified - use Monte Carlo dropout in practice)
            uncertainty = torch.std(prediction, dim=0, keepdim=True)
            
            # Generate corrective prompts
            uncertainty_np = uncertainty.cpu().numpy()[0, 0]
            prediction_np = prediction.cpu().numpy()[0, 0]
            
            corrective_points = self.generate_corrective_prompts(
                prediction_np,
                uncertainty_np,
                num_points=5,
                uncertainty_threshold=uncertainty_threshold
            )
            
            if len(corrective_points) == 0:
                break
            
            # Update prompts
            current_prompts = self._merge_prompts(current_prompts, corrective_points)
            prompt_history.append(current_prompts)
        
        return prediction, prompt_history
    
    def _merge_prompts(self, existing_prompts: Dict, new_points: np.ndarray) -> Dict:
        """Merge existing prompts with new corrective points"""
        # Implementation depends on prompt format
        # This is a simplified version
        merged = existing_prompts.copy()
        if 'points' in merged:
            merged['points'] = np.vstack([merged['points'], new_points])
        else:
            merged['points'] = new_points
        return merged


if __name__ == "__main__":
    # Test prompt generation
    print("Testing Prompt Generation...")
    
    # Create dummy mask
    mask = np.zeros((256, 256))
    mask[100:150, 100:150] = 1
    
    generator = PromptGenerator()
    
    # Test box generation
    box = generator.generate_box_from_mask(mask)
    print(f"Generated box: {box}")
    
    # Test point generation
    pos_points, neg_points = generator.generate_points_from_mask(mask, num_positive=5, num_negative=3)
    print(f"Positive points: {pos_points.shape}")
    print(f"Negative points: {neg_points.shape}")
    
    # Test batch generation
    batch_generator = BatchPromptGenerator()
    masks = torch.rand(2, 3, 256, 256) > 0.5
    boxes = batch_generator.generate_box_prompts(masks.float())
    print(f"Batch boxes shape: {boxes.shape}")
