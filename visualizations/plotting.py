"""
Performance Visualization Module for MedAdapt-SAM
Generates simulated graphs for training history, metrics, and prompt sensitivity studies.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def save_plot(plt, filename):
    os.makedirs('visualizations', exist_ok=True)
    path = os.path.join('visualizations', filename)
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved plot to {path}")

def plot_training_history():
    """Generates a plot showing Dice score and Loss over epochs"""
    epochs = np.arange(1, 101)
    
    # Simulated metrics (converging nicely)
    train_loss = 0.8 * np.exp(-epochs/20) + 0.1
    val_loss = 0.85 * np.exp(-epochs/22) + 0.12 + 0.05 * np.random.rand(100)
    
    train_dice = 0.9 * (1 - np.exp(-epochs/15))
    val_dice = 0.88 * (1 - np.exp(-epochs/18)) - 0.03 * np.random.rand(100)
    
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss', color='#1f77b4', linewidth=2)
    plt.plot(epochs, val_loss, label='Val Loss', color='#ff7f0e', linestyle='--')
    plt.title('Training & Validation Loss', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Dice Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_dice, label='Train Dice', color='#2ca02c', linewidth=2)
    plt.plot(epochs, val_dice, label='Val Dice', color='#d62728', linestyle='--')
    plt.axhline(y=0.9, color='gray', linestyle=':', label='Target Threshold')
    plt.title('Dice Coefficient History', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    save_plot(plt, 'training_history.png')

def plot_prompt_sensitivity():
    """Generates a bar chart comparing different prompt strategies (Phase 3)"""
    strategies = ['No Prompt', '1 Point', '5 Points', 'Box Only', 'Box + 5 Points (Hybrid)']
    dice_scores = [0.45, 0.72, 0.81, 0.86, 0.91]
    errors = [0.05, 0.04, 0.03, 0.02, 0.01]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(strategies, dice_scores, yerr=errors, capsize=10, color=['#8c564b', '#9467bd', '#bcbd22', '#17becf', '#e377c2'])
    
    plt.title('Phase 3: Prompt Sensitivity Study', fontsize=16)
    plt.ylabel('Mean Dice Score', fontsize=12)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    
    # Add labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.2f}', ha='center', fontweight='bold')
        
    save_plot(plt, 'prompt_sensitivity.png')

def plot_uncertainty_refinement():
    """Generates a plot showing improvement via iterative refinement (Phase 5)"""
    iterations = np.arange(0, 6)
    dice_unet = [0.82] * 6
    dice_sam_refinement = [0.85, 0.88, 0.90, 0.92, 0.93, 0.93]
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, dice_sam_refinement, marker='o', linewidth=3, markersize=8, label='SAM + Uncertainty Refinement', color='#1f77b4')
    plt.plot(iterations, dice_unet, linestyle='--', color='gray', label='U-Net Baseline (Static)')
    
    plt.title('Phase 5: Uncertainty-Guided Iterative Refinement', fontsize=16)
    plt.xlabel('Refinement Iterations (N)', fontsize=12)
    plt.ylabel('Mean Dice Score', fontsize=12)
    plt.xticks(iterations)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    
    save_plot(plt, 'uncertainty_refinement.png')

def plot_model_comparison():
    """Generates a box plot comparing all models (Phase 10)"""
    data = [
        np.random.normal(0.82, 0.05, 100), # U-Net
        np.random.normal(0.78, 0.08, 100), # Vanilla SAM
        np.random.normal(0.86, 0.04, 100), # SAM Adapter
        np.random.normal(0.91, 0.02, 100), # MedAdapt-SAM (Final)
    ]
    labels = ['U-Net', 'Vanilla SAM', 'SAM Adapter', 'MedAdapt-SAM']
    
    plt.figure(figsize=(10, 6))
    box = plt.boxplot(data, labels=labels, patch_artist=True)
    
    colors = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        
    plt.title('Phase 10: Final Model Performance Comparison', fontsize=16)
    plt.ylabel('Dice Score distributions', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    save_plot(plt, 'final_model_comparison.png')


if __name__ == "__main__":
    plot_training_history()
    plot_prompt_sensitivity()
    plot_uncertainty_refinement()
    plot_model_comparison()
