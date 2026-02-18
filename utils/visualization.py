"""
Visualization utilities for training, evaluation, and retrieval tasks.
Supports Task 1 (classification), Task 2 (report generation), and Task 3 (retrieval).
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import seaborn as sns
from pathlib import Path


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plot training curves from history dictionary.
    
    Args:
        history: Dictionary with training history
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss curves
    axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0].plot(history['val_loss'], label='Val Loss', color='red')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(history['train_acc'], label='Train Acc', color='blue')
    axes[1].plot(history['val_acc'], label='Val Acc', color='red')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate
    if 'learning_rates' in history and history['learning_rates']:
        axes[2].plot(history['learning_rates'], color='green')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'No learning rate data', 
                    ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Learning Rate')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                         save_path: Optional[str] = None):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel='Predicted',
           ylabel='Actual')
    
    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    ax.set_title('Confusion Matrix')
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc_score: float,
                  save_path: Optional[str] = None):
    """
    Plot ROC curve.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_score: Area under the curve
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_failure_cases(images: np.ndarray, true_labels: List[int], 
                      pred_labels: List[int], class_names: List[str],
                      save_path: Optional[str] = None):
    """
    Plot failure cases.
    
    Args:
        images: Array of images
        true_labels: Ground truth labels
        pred_labels: Predicted labels
        class_names: List of class names
        save_path: Path to save the figure
    """
    n_images = len(images)
    n_cols = min(4, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    for idx, ax in enumerate(axes):
        if idx < n_images:
            img = images[idx].squeeze()
            if img.max() > 1:  # Denormalize if needed
                img = img / 255.0
            
            ax.imshow(img, cmap='gray')
            ax.set_title(f'True: {class_names[true_labels[idx]]}\nPred: {class_names[pred_labels[idx]]}')
            ax.axis('off')
            
            # Color code border
            if true_labels[idx] != pred_labels[idx]:
                for spine in ax.spines.values():
                    spine.set_color('red')
                    spine.set_linewidth(2)
        else:
            ax.axis('off')
    
    plt.suptitle('Failure Cases (Red Border = Misclassification)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_retrieval_results(query_image: Optional[np.ndarray], 
                          results: List[Dict],
                          class_names: List[str],
                          query_text: Optional[str] = None,
                          save_path: Optional[str] = None,
                          show: bool = True):
    """
    Plot retrieval results for Task 3.
    
    Args:
        query_image: Query image (for image-to-image search)
        results: List of retrieved results with metadata
        class_names: List of class names
        query_text: Query text (for text-to-image search)
        save_path: Path to save the figure
        show: Whether to display the figure
        
    Returns:
        matplotlib figure
    """
    n_results = len(results)
    
    # Calculate grid dimensions
    if query_image is not None or query_text is not None:
        n_cols = min(5, n_results + 1)
        n_rows = (n_results + 1 + n_cols - 1) // n_cols
        total_plots = n_results + 1
    else:
        n_cols = min(4, n_results)
        n_rows = (n_results + n_cols - 1) // n_cols
        total_plots = n_results
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    
    # Handle different return types from plt.subplots
    if n_rows == 1 and n_cols == 1:
        # Single subplot
        axes = np.array([[axes]])
    elif n_rows == 1:
        # Single row, multiple columns
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        # Single column, multiple rows
        axes = axes.reshape(-1, 1)
    # else: already a 2D array
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    plot_idx = 0
    
    # Plot query
    if query_image is not None:
        # Denormalize query image
        if isinstance(query_image, np.ndarray):
            img_display = query_image.squeeze()
            if img_display.max() <= 1.0 and img_display.min() >= -1.0:
                img_display = (img_display * 0.5) + 0.5  # From [-1,1] to [0,1]
            if img_display.max() <= 1.0:
                img_display = (img_display * 255).astype(np.uint8)
            
            axes_flat[plot_idx].imshow(img_display, cmap='gray')
            axes_flat[plot_idx].set_title("Query Image", fontweight='bold', fontsize=10)
        axes_flat[plot_idx].axis('off')
        plot_idx += 1
        
    elif query_text is not None:
        axes_flat[plot_idx].text(0.5, 0.5, f"Query:\n{query_text}", 
                                ha='center', va='center', wrap=True, fontsize=9,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes_flat[plot_idx].axis('off')
        plot_idx += 1
    
    # Plot results
    for i, result in enumerate(results):
        if plot_idx >= len(axes_flat):
            break
            
        if 'image' in result and result['image'] is not None:
            img = result['image'].squeeze()
            if img.max() <= 1.0 and img.min() >= -1.0:
                img = (img * 0.5) + 0.5  # From [-1,1] to [0,1]
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            
            axes_flat[plot_idx].imshow(img, cmap='gray')
        else:
            axes_flat[plot_idx].text(0.5, 0.5, f"ID: {result.get('image_id', 'N/A')}", 
                                    ha='center', va='center')
        
        # Color code by label
        label = result.get('label', 0)
        label_color = 'green' if label == 0 else 'red'
        
        similarity = result.get('similarity', result.get('distance', 0))
        title = f"Rank {result.get('rank', i+1)}\n{result.get('label_name', 'Unknown')}\nSim: {similarity:.3f}"
        axes_flat[plot_idx].set_title(title, fontsize=8, color=label_color)
        axes_flat[plot_idx].axis('off')
        
        plot_idx += 1
    
    # Turn off unused subplots
    for i in range(plot_idx, len(axes_flat)):
        axes_flat[i].axis('off')
    
    # Add overall title
    if query_image is not None:
        plt.suptitle("Image-to-Image Search Results", fontsize=14, fontweight='bold')
    elif query_text is not None:
        plt.suptitle(f"Text-to-Image Search Results", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_precision_at_k(precision_dict: Dict[str, float], 
                       save_path: Optional[str] = None):
    """
    Plot precision@k results.
    
    Args:
        precision_dict: Dictionary with precision@k values (e.g., {'P@1': 0.8, 'P@5': 0.6})
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    k_values = list(precision_dict.keys())
    precision_values = list(precision_dict.values())
    
    bars = ax.bar(k_values, precision_values, color='skyblue', edgecolor='navy')
    
    # Add value labels on bars
    for bar, val in zip(bars, precision_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('k')
    ax.set_ylabel('Precision')
    ax.set_title('Precision@k for Image Retrieval')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_sample_reports(samples: List[Dict], 
                       figures_dir: Optional[Path] = None):
    """
    Plot sample reports for Task 2.
    
    Args:
        samples: List of sample report data
        figures_dir: Directory to save figures
    """
    for sample in samples:
        idx = sample.get('sample_id', 0)
        true_label = sample.get('true_label', 'Unknown')
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Show image if available
        if 'image' in sample:
            img = sample['image'].squeeze()
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            
            axes[0].imshow(img, cmap='gray')
            axes[0].set_title(f"Sample {idx}: {true_label}")
        axes[0].axis('off')
        
        # Show report
        prompt_comparison = sample.get('prompt_comparison', {})
        structured_result = prompt_comparison.get('structured', {})
        report_text = structured_result.get('report', 'No report generated')
        
        if len(report_text) > 500:
            report_text = report_text[:500] + "...\n[truncated]"
        
        axes[1].text(0.05, 0.95, 
                    f"MedGemma-4b-it Report:\n\n{report_text}",
                    transform=axes[1].transAxes,
                    wrap=True,
                    fontsize=9,
                    verticalalignment='top',
                    family='monospace')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if figures_dir:
            save_path = figures_dir / f'sample_{idx}_report.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close(fig)
    
    return True
