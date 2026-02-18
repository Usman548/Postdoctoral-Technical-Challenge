# Pneumonia Classification Model Evaluation Report

**Date:** 2026-02-17 15:06:58

## Model Configuration

- **Model Architecture:** vit-tiny
- **Model Path:** C:\Users\HP\Desktop\Alfsaisal University\7_day_challenge\models\saved\best_model_vit-tiny.pth
- **Test Samples:** 624

## Overall Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.8510 |
| Balanced Accuracy | 0.8124 |
| Precision | 0.8249 |
| Recall | 0.9667 |
| F1-Score | 0.8902 |
| AUC-ROC | 0.9223 |
| Cohen's Kappa | 0.6627 |
| Misclassification Rate | 0.1490 |

## Confusion Matrix

| | Predicted Normal | Predicted Pneumonia |
|-------|------------------|---------------------|
| Actual Normal | 154 | 80 |
| Actual Pneumonia | 13 | 377 |

## Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Normal | 0.9222 | 0.6581 | 0.7681 | 234.0 |
| Pneumonia | 0.8249 | 0.9667 | 0.8902 | 390.0 |

## Failure Analysis

- **Total Misclassified:** 93
- **False Positives:** 80 (34.19%)
- **False Negatives:** 13 (3.33%)

## Visualizations

The following visualizations have been generated:

- `figures/confusion_matrix.png` - Confusion matrix with counts and percentages
- `figures/roc_curve.png` - ROC curve with AUC
- `figures/confidence_distribution.png` - Confidence score distribution
- `figures/per_class_performance.png` - Per-class performance metrics
- `figures/failure_cases.png` - Failure case examples

## Model Strengths and Limitations

### Strengths
- Excellent discriminative ability (AUC > 0.9)
- Balanced precision and recall
- High overall accuracy

### Limitations
- High false positive rate, may lead to unnecessary follow-up

### Recommendations for Improvement
1. **Data Augmentation:** Apply more aggressive augmentation to improve generalization
2. **Class Weights:** Adjust class weights to handle imbalance
3. **Ensemble Methods:** Combine multiple models for better performance
4. **Transfer Learning:** Use pretrained models on larger medical datasets
5. **Threshold Tuning:** Optimize classification threshold for specific clinical needs
