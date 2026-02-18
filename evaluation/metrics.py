"""
Evaluation Metrics for Medical Image Segmentation
Includes Dice, IoU, Hausdorff Distance, Precision, Recall
"""

import torch
import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import Dict, List, Tuple


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate Dice Coefficient
    
    Args:
        pred: Predicted mask (B, C, H, W)
        target: Ground truth mask (B, C, H, W)
        smooth: Smoothing factor
        
    Returns:
        Dice score per class
    """
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean(dim=0)  # Average over batch


def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate IoU (Jaccard Index)
    
    Args:
        pred: Predicted mask (B, C, H, W)
        target: Ground truth mask (B, C, H, W)
        smooth: Smoothing factor
        
    Returns:
        IoU score per class
    """
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean(dim=0)


def precision_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Calculate Precision"""
    pred = (pred > 0.5).float()
    
    true_positive = (pred * target).sum(dim=(2, 3))
    predicted_positive = pred.sum(dim=(2, 3))
    
    precision = (true_positive + smooth) / (predicted_positive + smooth)
    return precision.mean(dim=0)


def recall_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Calculate Recall (Sensitivity)"""
    pred = (pred > 0.5).float()
    
    true_positive = (pred * target).sum(dim=(2, 3))
    actual_positive = target.sum(dim=(2, 3))
    
    recall = (true_positive + smooth) / (actual_positive + smooth)
    return recall.mean(dim=0)


def hausdorff_distance(pred: np.ndarray, target: np.ndarray, percentile: int = 100) -> float:
    """
    Calculate Hausdorff Distance
    
    Args:
        pred: Predicted binary mask (H, W)
        target: Ground truth binary mask (H, W)
        percentile: Percentile for robust HD (95 for HD95)
        
    Returns:
        Hausdorff distance
    """
    pred = (pred > 0.5).astype(np.uint8)
    target = target.astype(np.uint8)
    
    # Handle empty masks
    if pred.sum() == 0 or target.sum() == 0:
        return float('inf') if pred.sum() != target.sum() else 0.0
    
    # Compute distance transforms
    pred_dt = distance_transform_edt(1 - pred)
    target_dt = distance_transform_edt(1 - target)
    
    # Get surface points
    pred_surface = pred - np.logical_and(pred, distance_transform_edt(pred) > 1)
    target_surface = target - np.logical_and(target, distance_transform_edt(target) > 1)
    
    # Distances from pred surface to target
    pred_to_target = pred_dt[pred_surface > 0]
    
    # Distances from target surface to pred
    target_to_pred = target_dt[target_surface > 0]
    
    # Compute Hausdorff distance
    if percentile == 100:
        hd = max(pred_to_target.max(), target_to_pred.max())
    else:
        hd = max(
            np.percentile(pred_to_target, percentile),
            np.percentile(target_to_pred, percentile)
        )
    
    return float(hd)


def hausdorff_distance_95(pred: np.ndarray, target: np.ndarray) -> float:
    """Calculate 95th percentile Hausdorff Distance"""
    return hausdorff_distance(pred, target, percentile=95)


class MetricsCalculator:
    """Calculate all metrics for segmentation evaluation"""
    
    def __init__(self, num_classes: int = 3, class_names: List[str] = None):
        """
        Args:
            num_classes: Number of segmentation classes
            class_names: Names of classes (e.g., ['ET', 'TC', 'WT'])
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.dice_scores = []
        self.iou_scores = []
        self.precision_scores = []
        self.recall_scores = []
        self.hd_scores = []
        self.hd95_scores = []
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with new batch
        
        Args:
            pred: Predicted masks (B, C, H, W)
            target: Ground truth masks (B, C, H, W)
        """
        # Calculate tensor-based metrics
        dice = dice_coefficient(pred, target)
        iou = iou_score(pred, target)
        precision = precision_score(pred, target)
        recall = recall_score(pred, target)
        
        self.dice_scores.append(dice.cpu().numpy())
        self.iou_scores.append(iou.cpu().numpy())
        self.precision_scores.append(precision.cpu().numpy())
        self.recall_scores.append(recall.cpu().numpy())
        
        # Calculate Hausdorff distances (per sample)
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()
        
        batch_hd = []
        batch_hd95 = []
        
        for b in range(pred_np.shape[0]):
            sample_hd = []
            sample_hd95 = []
            
            for c in range(pred_np.shape[1]):
                hd = hausdorff_distance(pred_np[b, c], target_np[b, c])
                hd95 = hausdorff_distance_95(pred_np[b, c], target_np[b, c])
                
                sample_hd.append(hd)
                sample_hd95.append(hd95)
            
            batch_hd.append(sample_hd)
            batch_hd95.append(sample_hd95)
        
        self.hd_scores.extend(batch_hd)
        self.hd95_scores.extend(batch_hd95)
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics"""
        
        # Average over all batches
        dice_mean = np.mean(self.dice_scores, axis=0)
        iou_mean = np.mean(self.iou_scores, axis=0)
        precision_mean = np.mean(self.precision_scores, axis=0)
        recall_mean = np.mean(self.recall_scores, axis=0)
        
        # Filter out infinite values for HD
        hd_array = np.array(self.hd_scores)
        hd95_array = np.array(self.hd95_scores)
        
        hd_mean = np.array([
            np.mean(hd_array[np.isfinite(hd_array[:, c]), c]) 
            for c in range(self.num_classes)
        ])
        
        hd95_mean = np.array([
            np.mean(hd95_array[np.isfinite(hd95_array[:, c]), c]) 
            for c in range(self.num_classes)
        ])
        
        # Create results dictionary
        results = {}
        
        for i, class_name in enumerate(self.class_names):
            results[f'dice_{class_name}'] = float(dice_mean[i])
            results[f'iou_{class_name}'] = float(iou_mean[i])
            results[f'precision_{class_name}'] = float(precision_mean[i])
            results[f'recall_{class_name}'] = float(recall_mean[i])
            results[f'hd_{class_name}'] = float(hd_mean[i])
            results[f'hd95_{class_name}'] = float(hd95_mean[i])
        
        # Overall averages
        results['dice_mean'] = float(dice_mean.mean())
        results['iou_mean'] = float(iou_mean.mean())
        results['precision_mean'] = float(precision_mean.mean())
        results['recall_mean'] = float(recall_mean.mean())
        results['hd_mean'] = float(hd_mean.mean())
        results['hd95_mean'] = float(hd95_mean.mean())
        
        return results
    
    def print_results(self):
        """Print formatted results"""
        results = self.compute()
        
        print("\n" + "="*60)
        print("SEGMENTATION METRICS")
        print("="*60)
        
        for class_name in self.class_names:
            print(f"\n{class_name}:")
            print(f"  Dice:      {results[f'dice_{class_name}']:.4f}")
            print(f"  IoU:       {results[f'iou_{class_name}']:.4f}")
            print(f"  Precision: {results[f'precision_{class_name}']:.4f}")
            print(f"  Recall:    {results[f'recall_{class_name}']:.4f}")
            print(f"  HD:        {results[f'hd_{class_name}']:.4f}")
            print(f"  HD95:      {results[f'hd95_{class_name}']:.4f}")
        
        print(f"\nOverall Averages:")
        print(f"  Dice:      {results['dice_mean']:.4f}")
        print(f"  IoU:       {results['iou_mean']:.4f}")
        print(f"  Precision: {results['precision_mean']:.4f}")
        print(f"  Recall:    {results['recall_mean']:.4f}")
        print(f"  HD:        {results['hd_mean']:.4f}")
        print(f"  HD95:      {results['hd95_mean']:.4f}")
        print("="*60 + "\n")


if __name__ == "__main__":
    # Test metrics
    pred = torch.rand(4, 3, 256, 256)
    target = torch.randint(0, 2, (4, 3, 256, 256)).float()
    
    calculator = MetricsCalculator(num_classes=3, class_names=['ET', 'TC', 'WT'])
    calculator.update(pred, target)
    calculator.print_results()
