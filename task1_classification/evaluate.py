#!/usr/bin/env python3
"""
Evaluation script for pneumonia classification model.
Generates comprehensive metrics, visualizations, and failure case analysis.
Optimized for CPU execution.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                           confusion_matrix, roc_curve, auc, classification_report,
                           balanced_accuracy_score, cohen_kappa_score)
from pathlib import Path
import sys
import logging
import json
import argparse
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Local imports
from core.protocols import DatasetProvider
from data.data_loader import PneumoniaMNISTDataset
from models.cnn_model import create_model
from utils.logger import setup_logger

# Configure logging
logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# SOLID SRP: Metrics computation and visualization are separate responsibilities
# ---------------------------------------------------------------------------

class EvaluationMetricsCalculator:
    """Computes evaluation metrics from predictions. Single responsibility."""

    @staticmethod
    def compute(
        all_labels: np.ndarray,
        all_predictions: np.ndarray,
        all_probabilities: np.ndarray,
        all_confidence_scores: np.ndarray,
        class_names: List[str],
    ) -> Dict[str, Any]:
        """Return same metric dict as before (no behavior change)."""
        metrics = {}
        metrics["accuracy"] = float(accuracy_score(all_labels, all_predictions))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(all_labels, all_predictions))
        metrics["cohen_kappa"] = float(cohen_kappa_score(all_labels, all_predictions))
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average="binary", zero_division=0
        )
        metrics["precision"] = float(precision)
        metrics["recall"] = float(recall)
        metrics["f1_score"] = float(f1)
        class_report = classification_report(
            all_labels, all_predictions,
            target_names=class_names, output_dict=True, zero_division=0
        )
        metrics["classification_report"] = class_report
        fpr, tpr, _ = roc_curve(all_labels, all_probabilities[:, 1], pos_label=1)
        metrics["auc"] = float(auc(fpr, tpr))
        metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        cm = confusion_matrix(all_labels, all_predictions)
        metrics["confusion_matrix"] = cm.tolist()
        metrics["confusion_matrix_percent"] = (cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]).tolist()
        metrics["true_positives"] = int(cm[1, 1])
        metrics["true_negatives"] = int(cm[0, 0])
        metrics["false_positives"] = int(cm[0, 1])
        metrics["false_negatives"] = int(cm[1, 0])
        tp, tn, fp, fn = metrics["true_positives"], metrics["true_negatives"], metrics["false_positives"], metrics["false_negatives"]
        metrics["tpr"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0
        metrics["tnr"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0
        metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0
        metrics["fnr"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0
        metrics["false_positive_rate"] = metrics["fpr"]
        metrics["false_negative_rate"] = metrics["fnr"]
        metrics["sensitivity"] = metrics["tpr"]
        metrics["specificity"] = metrics["tnr"]
        metrics["mean_confidence"] = float(np.mean(all_confidence_scores))
        metrics["std_confidence"] = float(np.std(all_confidence_scores))
        metrics["num_misclassified"] = int(np.sum(all_labels != all_predictions))
        metrics["misclassification_rate"] = float(metrics["num_misclassified"] / len(all_labels))
        return metrics


class EvaluationFigureWriter:
    """Writes evaluation figures. Single responsibility; same outputs as before."""

    def __init__(self, figures_dir: Path):
        self.figures_dir = Path(figures_dir)

    def write_all(
        self,
        metrics: Dict[str, Any],
        all_labels: np.ndarray,
        all_predictions: np.ndarray,
        all_probabilities: np.ndarray,
        all_confidence_scores: np.ndarray,
        class_names: List[str],
    ) -> None:
        self._write_confusion_matrix(all_labels, all_predictions, class_names)
        self._write_roc_curve(all_labels, all_probabilities, metrics.get("auc"))
        self._write_confidence_distribution(all_labels, all_predictions, all_confidence_scores)
        self._write_per_class_performance(all_labels, all_predictions, class_names)

    def _write_confusion_matrix(
        self, all_labels: np.ndarray, all_predictions: np.ndarray, class_names: List[str]
    ) -> None:
        cm = confusion_matrix(all_labels, all_predictions)
        cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=axes[0])
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")
        axes[0].set_title("Confusion Matrix (Counts)")
        sns.heatmap(cm_percent, annot=True, fmt=".1%", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=axes[1])
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Actual")
        axes[1].set_title("Confusion Matrix (Percentages)")
        plt.tight_layout()
        plt.savefig(self.figures_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close()

    def _write_roc_curve(
        self, all_labels: np.ndarray, all_probabilities: np.ndarray, roc_auc: Optional[float] = None
    ) -> None:
        fpr, tpr, _ = roc_curve(all_labels, all_probabilities[:, 1], pos_label=1)
        if roc_auc is None:
            roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], "r--", linewidth=1, label="Random classifier (AUC = 0.5)")
        plt.fill_between(fpr, 0, tpr, alpha=0.1, color="blue")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        plt.text(0.05, 0.95, f"AUC = {roc_auc:.3f}", transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment="top", bbox=props)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
        plt.close()

    def _write_confidence_distribution(
        self,
        all_labels: np.ndarray,
        all_predictions: np.ndarray,
        all_confidence_scores: np.ndarray,
    ) -> None:
        correct_mask = all_labels == all_predictions
        incorrect_mask = ~correct_mask
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(all_confidence_scores[correct_mask], bins=20, alpha=0.7, label="Correct", color="green", density=True)
        axes[0].hist(all_confidence_scores[incorrect_mask], bins=20, alpha=0.7, label="Incorrect", color="red", density=True)
        axes[0].set_xlabel("Confidence Score")
        axes[0].set_ylabel("Density")
        axes[0].set_title("Confidence Distribution")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        data = [all_confidence_scores[correct_mask], all_confidence_scores[incorrect_mask]]
        bp = axes[1].boxplot(data, tick_labels=["Correct", "Incorrect"], patch_artist=True)
        bp["boxes"][0].set_facecolor("lightgreen")
        bp["boxes"][1].set_facecolor("lightcoral")
        axes[1].set_ylabel("Confidence Score")
        axes[1].set_title("Confidence by Prediction Correctness")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "confidence_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()

    def _write_per_class_performance(
        self,
        all_labels: np.ndarray,
        all_predictions: np.ndarray,
        class_names: List[str],
    ) -> None:
        report = classification_report(
            all_labels, all_predictions, target_names=class_names, output_dict=True
        )
        classes = list(class_names)
        precision = [report[cls]["precision"] for cls in classes]
        recall = [report[cls]["recall"] for cls in classes]
        f1 = [report[cls]["f1-score"] for cls in classes]
        x = np.arange(len(classes))
        width = 0.25
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width, precision, width, label="Precision", color="skyblue")
        rects2 = ax.bar(x, recall, width, label="Recall", color="lightcoral")
        rects3 = ax.bar(x + width, f1, width, label="F1-Score", color="lightgreen")
        ax.set_xlabel("Class")
        ax.set_ylabel("Score")
        ax.set_title("Per-Class Performance Metrics")
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        for rects in [rects1, rects2, rects3]:
            for rect in rects:
                h = rect.get_height()
                ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "per_class_performance.png", dpi=150, bbox_inches="tight")
        plt.close()


@dataclass
class EvaluationConfig:
    """Evaluation configuration dataclass."""
    
    # Model parameters
    model_path: str
    model_name: str = 'custom'
    
    # Data parameters
    batch_size: int = 64
    augment: bool = False
    random_seed: int = 42
    
    # Evaluation parameters
    device: str = 'cpu'
    num_workers: int = 0
    pin_memory: bool = False
    
    # Visualization parameters
    num_failure_cases: int = 8
    confidence_threshold: float = 0.5
    
    # Output paths
    output_dir: str = 'reports/task1'
    figures_dir: str = 'reports/task1/figures'
    
    def __post_init__(self):
        """Validate configuration."""
        assert Path(self.model_path).exists(), f"Model path does not exist: {self.model_path}"
        assert self.batch_size > 0, "Batch size must be positive"
        assert 0 <= self.confidence_threshold <= 1, "Confidence threshold must be between 0 and 1"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


class Evaluator:
    """
    Comprehensive evaluator for pneumonia classification model.
    Depends on DatasetProvider when dataset is injected (DIP); same API and behavior.
    """

    def __init__(
        self,
        config: Union[Dict[str, Any], EvaluationConfig, str],
        dataset: Optional[DatasetProvider] = None,
    ):
        """
        Initialize evaluator.

        Args:
            config: Evaluation configuration (dict, EvaluationConfig, or path to JSON)
            dataset: Optional DatasetProvider; if None, PneumoniaMNISTDataset is used (backward compatible).
        """
        # Load configuration
        if isinstance(config, str):
            with open(config, "r") as f:
                config_dict = json.load(f)
            self.config = EvaluationConfig(**config_dict)
        elif isinstance(config, dict):
            self.config = EvaluationConfig(**config)
        elif isinstance(config, EvaluationConfig):
            self.config = config
        else:
            raise ValueError(f"Invalid config type: {type(config)}")

        self.device = torch.device("cpu")
        logger.info(f"Using device: {self.device}")

        self._setup_directories()
        self._setup_data(dataset)
        
        # Load model
        self._setup_model()
        
        # Initialize storage for predictions
        self._initialize_storage()
        # SOLID SRP: delegate figure writing to dedicated class
        self._figure_writer = EvaluationFigureWriter(self.figures_dir)

        logger.info("Evaluator initialization complete")

    def _setup_directories(self) -> None:
        """Create output directories."""
        self.output_dir = Path(self.config.output_dir)
        self.figures_dir = Path(self.config.figures_dir)
        for directory in [self.output_dir, self.figures_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
    
    def _setup_data(self, dataset: Optional[DatasetProvider] = None) -> None:
        """Load and prepare test dataset. Uses injected dataset if provided (DIP)."""
        logger.info("Loading test dataset...")
        try:
            if dataset is None:
                dataset = PneumoniaMNISTDataset(
                    batch_size=self.config.batch_size,
                    augment=self.config.augment,
                    random_seed=self.config.random_seed,
                    num_workers=self.config.num_workers,
                    pin_memory=self.config.pin_memory,
                )
            self.dataset = dataset
            _, _, self.test_loader = self.dataset.get_dataloaders()
            self.class_names = self.dataset.get_class_names()
            self.num_classes = len(self.class_names)
            logger.info("Test dataset loaded:")
            logger.info(f"  - Test samples: {len(self.test_loader.dataset)}")
            logger.info(f"  - Class names: {self.class_names}")
            logger.info(f"  - Batch size: {self.config.batch_size}")
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise
    
    def _setup_model(self) -> None:
        """Load trained model from checkpoint."""
        logger.info(f"Loading model from {self.config.model_path}")
        
        try:
            # Create model architecture
            self.model = create_model(
                model_name=self.config.model_name,
                num_classes=self.num_classes,
                pretrained=False
            )
            
            # Load checkpoint
            checkpoint = torch.load(
                self.config.model_path,
                map_location=self.device
            )
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint.get('epoch', 'unknown')
                val_acc = checkpoint.get('val_acc', 'unknown')
                logger.info(f"Loaded checkpoint from epoch {epoch} with val_acc: {val_acc}")
            else:
                # Assume direct state dict
                self.model.load_state_dict(checkpoint)
                logger.info("Loaded model state dict directly")
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Log model info
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Model loaded successfully:")
            logger.info(f"  - Architecture: {self.config.model_name}")
            logger.info(f"  - Total parameters: {total_params:,}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _initialize_storage(self) -> None:
        """Initialize storage arrays for predictions."""
        self.all_labels: List[int] = []
        self.all_predictions: List[int] = []
        self.all_probabilities: List[np.ndarray] = []
        self.all_images: List[np.ndarray] = []
        self.all_confidence_scores: List[float] = []
        
        # For per-sample tracking
        self.sample_indices: List[int] = []
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Run comprehensive evaluation pipeline.
        
        Returns:
            Dictionary containing all evaluation results
        """
        logger.info("=" * 60)
        logger.info("STARTING MODEL EVALUATION")
        logger.info("=" * 60)
        
        # Run inference on test set
        self._run_inference()
        
        # Convert lists to numpy arrays
        self._convert_to_arrays()
        
        # Calculate all metrics
        metrics = self._calculate_metrics()

        # Generate visualizations (SRP: delegated to EvaluationFigureWriter)
        self._generate_visualizations(metrics)
        
        # Analyze failure cases
        failure_analysis = self._analyze_failure_cases()
        
        # Save results
        self._save_results(metrics, failure_analysis)
        
        # Log summary
        self._log_summary(metrics)
        
        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        return {**metrics, **failure_analysis}
    
    def _run_inference(self) -> None:
        """Run model inference on test set."""
        logger.info("Running inference on test set...")
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(self.test_loader, 
                                                             desc="Evaluating")):
                # Move to CPU
                images = images.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                confidence_scores, predicted = torch.max(probabilities, dim=1)
                
                # Store results
                self.all_labels.extend(labels.cpu().numpy())
                self.all_predictions.extend(predicted.cpu().numpy())
                self.all_probabilities.extend(probabilities.cpu().numpy())
                self.all_images.extend(images.cpu().numpy())
                self.all_confidence_scores.extend(confidence_scores.cpu().numpy())
                
                # Store sample indices
                start_idx = batch_idx * self.config.batch_size
                self.sample_indices.extend(
                    range(start_idx, start_idx + len(images))
                )
                
                # Log progress periodically
                if batch_idx % 50 == 0 and batch_idx > 0:
                    logger.debug(f"Processed {batch_idx * self.config.batch_size} samples")
        
        logger.info(f"Inference completed. Processed {len(self.all_labels)} samples.")
    
    def _convert_to_arrays(self) -> None:
        """Convert lists to numpy arrays for efficient computation."""
        self.all_labels = np.array(self.all_labels)
        self.all_predictions = np.array(self.all_predictions)
        self.all_probabilities = np.array(self.all_probabilities)
        self.all_images = np.array(self.all_images)
        self.all_confidence_scores = np.array(self.all_confidence_scores)
        self.sample_indices = np.array(self.sample_indices)
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Delegate to EvaluationMetricsCalculator (SRP). Same return value as before."""
        logger.info("Calculating evaluation metrics...")
        return EvaluationMetricsCalculator.compute(
            self.all_labels,
            self.all_predictions,
            self.all_probabilities,
            self.all_confidence_scores,
            self.class_names,
        )
    
    def _generate_visualizations(self, metrics: Dict[str, Any]) -> None:
        """Delegate to EvaluationFigureWriter (SRP). Same figures as before."""
        logger.info("Generating visualizations...")
        self._figure_writer.write_all(
            metrics,
            self.all_labels,
            self.all_predictions,
            self.all_probabilities,
            self.all_confidence_scores,
            self.class_names,
        )
        logger.info(f"Visualizations saved to {self.figures_dir}")

    def _analyze_failure_cases(self) -> Dict[str, Any]:
        """
        Analyze and visualize failure cases.
        
        Returns:
            Dictionary with failure case analysis
        """
        logger.info("Analyzing failure cases...")
        
        # Find misclassified indices
        misclassified_mask = self.all_labels != self.all_predictions
        misclassified_indices = np.where(misclassified_mask)[0]
        
        if len(misclassified_indices) == 0:
            logger.info("No misclassified examples found!")
            return {'failure_cases': [], 'num_failures': 0}
        
        # Separate false positives and false negatives
        false_positives = np.where((self.all_labels == 0) & (self.all_predictions == 1))[0]
        false_negatives = np.where((self.all_labels == 1) & (self.all_predictions == 0))[0]
        
        failure_analysis = {
            'num_false_positives': len(false_positives),
            'num_false_negatives': len(false_negatives),
            'false_positive_rate': len(false_positives) / np.sum(self.all_labels == 0) if np.sum(self.all_labels == 0) > 0 else 0,
            'false_negative_rate': len(false_negatives) / np.sum(self.all_labels == 1) if np.sum(self.all_labels == 1) > 0 else 0,
            'failure_cases': []
        }
        
        # Select failure cases for visualization
        selected_indices = self._select_failure_cases(
            false_positives, false_negatives, misclassified_indices
        )
        
        # Visualize selected failure cases
        self._plot_failure_cases(selected_indices)
        
        # Collect detailed information for each failure case
        for idx in selected_indices:
            case_info = {
                'sample_index': int(self.sample_indices[idx]),
                'true_label': int(self.all_labels[idx]),
                'true_class': self.class_names[self.all_labels[idx]],
                'predicted_label': int(self.all_predictions[idx]),
                'predicted_class': self.class_names[self.all_predictions[idx]],
                'confidence': float(self.all_confidence_scores[idx]),
                'probabilities': self.all_probabilities[idx].tolist(),
                'error_type': 'false_positive' if (self.all_labels[idx] == 0 and 
                                                   self.all_predictions[idx] == 1) else 'false_negative'
            }
            failure_analysis['failure_cases'].append(case_info)
        
        logger.info(f"Found {len(misclassified_indices)} misclassified samples "
                   f"({len(false_positives)} FP, {len(false_negatives)} FN)")
        
        return failure_analysis
    
    def _select_failure_cases(self, false_positives: np.ndarray, 
                              false_negatives: np.ndarray,
                              misclassified_indices: np.ndarray) -> List[int]:
        """
        Select representative failure cases for visualization.
        
        Args:
            false_positives: Array of false positive indices
            false_negatives: Array of false negative indices
            misclassified_indices: Array of all misclassified indices
            
        Returns:
            List of selected indices
        """
        num_examples = self.config.num_failure_cases
        selected_indices = []
        
        # Take up to num_examples//2 from each category
        n_fp = min(len(false_positives), num_examples // 2)
        n_fn = min(len(false_negatives), num_examples // 2)
        
        if n_fp > 0:
            # Select false positives with highest confidence (most confident mistakes)
            fp_confidences = self.all_confidence_scores[false_positives]
            top_fp_indices = false_positives[np.argsort(fp_confidences)[-n_fp:]]
            selected_indices.extend(top_fp_indices.tolist())
        
        if n_fn > 0:
            # Select false negatives with highest confidence
            fn_confidences = self.all_confidence_scores[false_negatives]
            top_fn_indices = false_negatives[np.argsort(fn_confidences)[-n_fn:]]
            selected_indices.extend(top_fn_indices.tolist())
        
        # If we need more, add random misclassified samples
        if len(selected_indices) < num_examples:
            remaining = num_examples - len(selected_indices)
            other_indices = [i for i in misclassified_indices 
                           if i not in selected_indices]
            if len(other_indices) > 0:
                np.random.seed(self.config.random_seed)
                additional = np.random.choice(
                    other_indices, 
                    min(remaining, len(other_indices)), 
                    replace=False
                )
                selected_indices.extend(additional.tolist())
        
        return selected_indices[:num_examples]
    
    def _plot_failure_cases(self, selected_indices: List[int]) -> None:
        """
        Plot failure cases visualization.
        
        Args:
            selected_indices: List of indices to visualize
        """
        if not selected_indices:
            return
        
        # Calculate grid size
        n_cols = min(4, len(selected_indices))
        n_rows = (len(selected_indices) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, 
                                 figsize=(4 * n_cols, 4 * n_rows))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        
        for idx, ax in enumerate(axes):
            if idx < len(selected_indices):
                i = selected_indices[idx]
                
                # Get image and denormalize (assuming normalization was [-1, 1])
                img = self.all_images[i].squeeze()
                img = (img * 0.5) + 0.5  # Denormalize to [0, 1]
                img = np.clip(img, 0, 1)
                
                true_label = self.class_names[self.all_labels[i]]
                pred_label = self.class_names[self.all_predictions[i]]
                confidence = self.all_confidence_scores[i]
                
                ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                ax.set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.3f}',
                            fontsize=10)
                ax.axis('off')
                
                # Color code border based on error type
                if self.all_labels[i] == 0:  # False positive
                    for spine in ax.spines.values():
                        spine.set_color('red')
                        spine.set_linewidth(3)
                else:  # False negative
                    for spine in ax.spines.values():
                        spine.set_color('orange')
                        spine.set_linewidth(3)
            else:
                ax.axis('off')
        
        plt.suptitle('Failure Case Analysis (Red: False Positive, Orange: False Negative)', 
                    fontsize=14)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'failure_cases.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_results(self, metrics: Dict[str, Any], 
                      failure_analysis: Dict[str, Any]) -> None:
        """
        Save evaluation results to files.
        
        Args:
            metrics: Dictionary of evaluation metrics
            failure_analysis: Dictionary of failure case analysis
        """
        # Combine results
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.to_dict(),
            'metrics': metrics,
            'failure_analysis': failure_analysis
        }
        
        # Save as JSON
        json_path = self.output_dir / 'evaluation_results.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save as markdown report
        self._save_markdown_report(metrics, failure_analysis)
        
        # Save as CSV for easy viewing
        self._save_csv_summary(metrics)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def _save_markdown_report(self, metrics: Dict[str, Any], 
                              failure_analysis: Dict[str, Any]) -> None:
        """
        Save evaluation report in markdown format.
        
        Args:
            metrics: Dictionary of evaluation metrics
            failure_analysis: Dictionary of failure case analysis
        """
        report_path = self.output_dir / 'task1_classification_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Pneumonia Classification Model Evaluation Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Model configuration
            f.write("## Model Configuration\n\n")
            f.write(f"- **Model Architecture:** {self.config.model_name}\n")
            f.write(f"- **Model Path:** {self.config.model_path}\n")
            f.write(f"- **Test Samples:** {len(self.all_labels)}\n\n")
            
            # Overall metrics
            f.write("## Overall Performance Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Accuracy | {metrics['accuracy']:.4f} |\n")
            f.write(f"| Balanced Accuracy | {metrics['balanced_accuracy']:.4f} |\n")
            f.write(f"| Precision | {metrics['precision']:.4f} |\n")
            f.write(f"| Recall | {metrics['recall']:.4f} |\n")
            f.write(f"| F1-Score | {metrics['f1_score']:.4f} |\n")
            f.write(f"| AUC-ROC | {metrics['auc']:.4f} |\n")
            f.write(f"| Cohen's Kappa | {metrics['cohen_kappa']:.4f} |\n")
            f.write(f"| Misclassification Rate | {metrics['misclassification_rate']:.4f} |\n\n")
            
            # Confusion matrix
            f.write("## Confusion Matrix\n\n")
            f.write("| | Predicted Normal | Predicted Pneumonia |\n")
            f.write("|-------|------------------|---------------------|\n")
            f.write(f"| Actual Normal | {metrics['true_negatives']} | {metrics['false_positives']} |\n")
            f.write(f"| Actual Pneumonia | {metrics['false_negatives']} | {metrics['true_positives']} |\n\n")
            
            # Per-class metrics
            f.write("## Per-Class Performance\n\n")
            f.write("| Class | Precision | Recall | F1-Score | Support |\n")
            f.write("|-------|-----------|--------|----------|---------|\n")
            for class_name in self.class_names:
                class_metrics = metrics['classification_report'][class_name]
                f.write(f"| {class_name} | {class_metrics['precision']:.4f} | "
                       f"{class_metrics['recall']:.4f} | {class_metrics['f1-score']:.4f} | "
                       f"{class_metrics['support']} |\n")
            f.write("\n")
            
            # Failure analysis
            f.write("## Failure Analysis\n\n")
            f.write(f"- **Total Misclassified:** {metrics['num_misclassified']}\n")
            f.write(f"- **False Positives:** {failure_analysis.get('num_false_positives', 0)} "
                   f"({failure_analysis.get('false_positive_rate', 0):.2%})\n")
            f.write(f"- **False Negatives:** {failure_analysis.get('num_false_negatives', 0)} "
                   f"({failure_analysis.get('false_negative_rate', 0):.2%})\n\n")
            
            # Visualizations
            f.write("## Visualizations\n\n")
            f.write("The following visualizations have been generated:\n\n")
            f.write("- `figures/confusion_matrix.png` - Confusion matrix with counts and percentages\n")
            f.write("- `figures/roc_curve.png` - ROC curve with AUC\n")
            f.write("- `figures/confidence_distribution.png` - Confidence score distribution\n")
            f.write("- `figures/per_class_performance.png` - Per-class performance metrics\n")
            f.write("- `figures/failure_cases.png` - Failure case examples\n\n")
            
            # Model strengths and limitations
            f.write("## Model Strengths and Limitations\n\n")
            f.write("### Strengths\n")
            if metrics['auc'] > 0.9:
                f.write("- Excellent discriminative ability (AUC > 0.9)\n")
            if metrics['precision'] > 0.8 and metrics['recall'] > 0.8:
                f.write("- Balanced precision and recall\n")
            if metrics['accuracy'] > 0.85:
                f.write("- High overall accuracy\n")
            
            f.write("\n### Limitations\n")
            # Use fpr from metrics
            if metrics['fpr'] > 0.2:
                f.write("- High false positive rate, may lead to unnecessary follow-up\n")
            if metrics['fnr'] > 0.2:
                f.write("- High false negative rate, risk of missing pneumonia cases\n")
            if metrics['precision'] < 0.7:
                f.write("- Low precision, many false alarms\n")
            if metrics['recall'] < 0.7:
                f.write("- Low recall, misses many pneumonia cases\n")
            
            f.write("\n### Recommendations for Improvement\n")
            f.write("1. **Data Augmentation:** Apply more aggressive augmentation to improve generalization\n")
            f.write("2. **Class Weights:** Adjust class weights to handle imbalance\n")
            f.write("3. **Ensemble Methods:** Combine multiple models for better performance\n")
            f.write("4. **Transfer Learning:** Use pretrained models on larger medical datasets\n")
            f.write("5. **Threshold Tuning:** Optimize classification threshold for specific clinical needs\n")
    
    def _save_csv_summary(self, metrics: Dict[str, Any]) -> None:
        """Save metrics summary as CSV."""
        csv_path = self.output_dir / 'metrics_summary.csv'
        
        # Flatten metrics for CSV
        flat_metrics = {
            'accuracy': metrics['accuracy'],
            'balanced_accuracy': metrics['balanced_accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'auc': metrics['auc'],
            'cohen_kappa': metrics['cohen_kappa'],
            'true_positives': metrics['true_positives'],
            'true_negatives': metrics['true_negatives'],
            'false_positives': metrics['false_positives'],
            'false_negatives': metrics['false_negatives'],
            'fpr': metrics['fpr'],
            'fnr': metrics['fnr'],
            'num_misclassified': metrics['num_misclassified']
        }
        
        df = pd.DataFrame([flat_metrics])
        df.to_csv(csv_path, index=False)
    
    def _log_summary(self, metrics: Dict[str, Any]) -> None:
        """Log evaluation summary."""
        logger.info("\n" + "=" * 50)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Test Samples: {len(self.all_labels)}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"AUC: {metrics['auc']:.4f}")
        logger.info(f"Misclassified: {metrics['num_misclassified']} "
                   f"({metrics['misclassification_rate']:.2%})")
        logger.info("=" * 50)


def main():
    """Main entry point for evaluation script."""
    
    parser = argparse.ArgumentParser(
        description='Evaluate pneumonia classifier (CPU-optimized)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to saved model weights')
    parser.add_argument('--model_name', type=str, default='custom',
                       choices=['custom', 'resnet18', 'efficientnet-b0', 'vit-tiny'],
                       help='Model architecture')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--num_failure_cases', type=int, default=8,
                       help='Number of failure cases to visualize')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='reports/task1',
                       help='Directory for output files')
    
    # Configuration file
    parser.add_argument('--config', type=str,
                       help='Path to evaluation configuration JSON file')
    
    args = parser.parse_args()
    
    # If config file is provided, use it
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        evaluator = Evaluator(args.config)
    else:
        # Create config from command line arguments
        config = {
            'model_path': args.model_path,
            'model_name': args.model_name,
            'batch_size': args.batch_size,
            'num_failure_cases': args.num_failure_cases,
            'output_dir': args.output_dir,
            'figures_dir': str(Path(args.output_dir) / 'figures')
        }
        evaluator = Evaluator(config)
    
    # Run evaluation
    try:
        results = evaluator.evaluate()
        logger.info(f"\n✅ Evaluation completed successfully!")
        logger.info(f"📊 Results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        logger.exception("Detailed error trace:")
        sys.exit(1)


if __name__ == '__main__':
    main()
