# Pneumonia Classification Model Evaluation Report

**Date:** 2026-02-19 01:14:49

## Model Configuration

- **Model Architecture:** custom
- **Model Path:** models/saved/best_model_custom.pth
- **Test Samples:** 624

## Overall Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.8654 |
| Balanced Accuracy | 0.8222 |
| Precision | 0.8255 |
| Recall | 0.9949 |
| F1-Score | 0.9023 |
| AUC-ROC | 0.9632 |
| Cohen's Kappa | 0.6917 |
| Misclassification Rate | 0.1346 |

## Confusion Matrix

| | Predicted Normal | Predicted Pneumonia |
|-------|------------------|---------------------|
| Actual Normal | 152 | 82 |
| Actual Pneumonia | 2 | 388 |

## Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Normal | 0.9870 | 0.6496 | 0.7835 | 234.0 |
| Pneumonia | 0.8255 | 0.9949 | 0.9023 | 390.0 |

## Failure Analysis

- **Total Misclassified:** 84
- **False Positives:** 82 (35.04%)
- **False Negatives:** 2 (0.51%)

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
