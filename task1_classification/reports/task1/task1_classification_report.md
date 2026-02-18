# Pneumonia Classification Report (CPU)

**Model:** custom
**Date:** 2026-02-17 21:41:59

## Configuration (CPU Optimized)

- Model: custom
- Batch Size: 16 (reduced for CPU)
- Epochs: 10
- Learning Rate: 0.001
- Augmentation: True
- CPU Threads: 4

## Test Set Results

- Accuracy: 0.8429
- AUC: 0.9618
- Misclassified: 98/624 (15.71%)

## Confusion Matrix

| | Predicted Normal | Predicted Pneumonia |
|---|---|---|
| Actual Normal | 139 | 95 |
| Actual Pneumonia | 3 | 387 |
