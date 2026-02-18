"""
Metrics calculation utilities for medical image classification.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    cohen_kappa_score, matthews_corrcoef
)


class MedicalClassificationMetrics:
    """Calculate classification metrics for medical images."""
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 y_prob: Optional[np.ndarray] = None,
                 class_names: Optional[List[str]] = None):
        """
        Initialize metrics calculator.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            class_names: Names of classes
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_prob = np.array(y_prob) if y_prob is not None else None
        self.class_names = class_names or ['Normal', 'Pneumonia']
        
        # Get confusion matrix values
        tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred, labels=[0, 1]).ravel()
        self.tn, self.fp, self.fn, self.tp = tn, fp, fn, tp
    
    def get_all_metrics(self) -> Dict[str, float]:
        """Calculate all metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = float(accuracy_score(self.y_true, self.y_pred))
        metrics['precision'] = float(precision_score(self.y_true, self.y_pred, zero_division=0))
        metrics['recall'] = float(recall_score(self.y_true, self.y_pred, zero_division=0))
        metrics['f1_score'] = float(f1_score(self.y_true, self.y_pred, zero_division=0))
        
        # Confusion matrix metrics
        metrics['true_positives'] = int(self.tp)
        metrics['true_negatives'] = int(self.tn)
        metrics['false_positives'] = int(self.fp)
        metrics['false_negatives'] = int(self.fn)
        
        # Rates
        metrics['sensitivity'] = float(self.tp / (self.tp + self.fn)) if (self.tp + self.fn) > 0 else 0.0
        metrics['specificity'] = float(self.tn / (self.tn + self.fp)) if (self.tn + self.fp) > 0 else 0.0
        metrics['fpr'] = float(self.fp / (self.fp + self.tn)) if (self.fp + self.tn) > 0 else 0.0
        metrics['fnr'] = float(self.fn / (self.fn + self.tp)) if (self.fn + self.tp) > 0 else 0.0
        
        # Predictive values
        metrics['ppv'] = float(self.tp / (self.tp + self.fp)) if (self.tp + self.fp) > 0 else 0.0
        metrics['npv'] = float(self.tn / (self.tn + self.fn)) if (self.tn + self.fn) > 0 else 0.0
        
        # Statistical metrics
        metrics['cohen_kappa'] = float(cohen_kappa_score(self.y_true, self.y_pred))
        metrics['matthews_corrcoef'] = float(matthews_corrcoef(self.y_true, self.y_pred))
        
        # Probabilistic metrics
        if self.y_prob is not None:
            try:
                metrics['roc_auc'] = float(roc_auc_score(self.y_true, self.y_prob[:, 1]))
            except:
                metrics['roc_auc'] = 0.5
        
        # Medical metrics
        if metrics['specificity'] < 1:
            metrics['positive_lr'] = float(metrics['sensitivity'] / (1 - metrics['specificity'])) if (1 - metrics['specificity']) > 0 else float('inf')
        else:
            metrics['positive_lr'] = float('inf')
            
        if metrics['sensitivity'] < 1:
            metrics['negative_lr'] = float((1 - metrics['sensitivity']) / metrics['specificity']) if metrics['specificity'] > 0 else float('inf')
        else:
            metrics['negative_lr'] = 0.0
        
        metrics['youden_index'] = metrics['sensitivity'] + metrics['specificity'] - 1
        
        return metrics
    
    def get_per_class_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate per-class metrics."""
        report = classification_report(
            self.y_true, self.y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        per_class = {}
        for class_name in self.class_names:
            if class_name in report:
                per_class[class_name] = {
                    'precision': float(report[class_name]['precision']),
                    'recall': float(report[class_name]['recall']),
                    'f1_score': float(report[class_name]['f1-score']),
                    'support': int(report[class_name]['support'])
                }
        
        return per_class
    
    def generate_report(self) -> str:
        """Generate formatted text report."""
        metrics = self.get_all_metrics()
        per_class = self.get_per_class_metrics()
        
        lines = []
        lines.append("=" * 60)
        lines.append("CLASSIFICATION METRICS REPORT")
        lines.append("=" * 60)
        
        # Confusion Matrix
        lines.append("\nCONFUSION MATRIX:")
        lines.append("-" * 40)
        lines.append(f"{'':20} {'Predicted':>20}")
        lines.append(f"{'':20} {'Normal':>10} {'Pneumonia':>10}")
        lines.append("-" * 40)
        lines.append(f"{'Actual Normal':20} {metrics['true_negatives']:10d} {metrics['false_positives']:10d}")
        lines.append(f"{'Actual Pneumonia':20} {metrics['false_negatives']:10d} {metrics['true_positives']:10d}")
        
        # Overall Metrics
        lines.append("\nOVERALL METRICS:")
        lines.append("-" * 40)
        lines.append(f"Accuracy:             {metrics['accuracy']:.4f}")
        lines.append(f"Precision:            {metrics['precision']:.4f}")
        lines.append(f"Recall:               {metrics['recall']:.4f}")
        lines.append(f"F1-Score:             {metrics['f1_score']:.4f}")
        lines.append(f"Sensitivity:          {metrics['sensitivity']:.4f}")
        lines.append(f"Specificity:          {metrics['specificity']:.4f}")
        
        if 'roc_auc' in metrics:
            lines.append(f"ROC-AUC:              {metrics['roc_auc']:.4f}")
        
        lines.append(f"Cohen's Kappa:        {metrics['cohen_kappa']:.4f}")
        lines.append(f"Youden's Index:       {metrics['youden_index']:.4f}")
        
        # Per-Class Metrics
        lines.append("\nPER-CLASS METRICS:")
        lines.append("-" * 40)
        for class_name, class_metrics in per_class.items():
            lines.append(f"\n{class_name}:")
            lines.append(f"  Precision: {class_metrics['precision']:.4f}")
            lines.append(f"  Recall:    {class_metrics['recall']:.4f}")
            lines.append(f"  F1-Score:  {class_metrics['f1_score']:.4f}")
            lines.append(f"  Support:   {class_metrics['support']}")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)


def calculate_metrics(y_true, y_pred, y_prob=None, class_names=None):
    """Convenience function to calculate all metrics."""
    calculator = MedicalClassificationMetrics(y_true, y_pred, y_prob, class_names)
    return calculator.get_all_metrics()


def print_metrics_report(y_true, y_pred, y_prob=None, class_names=None):
    """Print formatted metrics report."""
    calculator = MedicalClassificationMetrics(y_true, y_pred, y_prob, class_names)
    print(calculator.generate_report())


# Alias for backward compatibility
compute_classification_metrics = calculate_metrics
