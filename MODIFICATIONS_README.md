# UNet 640x640 Training with PR Curves and Best Model Tracking

This document explains the comprehensive modifications made to the UNet training code to support:
- Fixed 640x640 input size during training with original dimension evaluation
- Class-wise mAP@IoU0.5 calculation and PR curve plotting  
- Best model tracking based on lowest validation loss
- Comprehensive metrics reporting and visualization

## Key Features Added

### 1. PR Curve Plotting & Class-wise mAP
- **Class-wise mAP@IoU0.5**: Calculates mAP at IoU threshold 0.5 for each class
- **PR Curves**: Generates precision-recall curves for each class
- **Average mAP**: Computes overall mAP across all classes
- **TensorBoard Integration**: All metrics logged to TensorBoard
- **Automatic Plotting**: PR curves saved as PNG files during training

### 2. Best Model Tracking
- **Validation Loss Monitoring**: Tracks lowest validation loss across epochs
- **Automatic Saving**: Best model automatically saved as `best_model.pth`
- **Comprehensive State**: Includes epoch, metrics, optimizer state, and class performance
- **Best Model Analysis**: Dedicated script for analyzing the best model

### 3. Enhanced Visualization
- **Epoch-level PR Curves**: Generated at each epoch
- **Class-specific Plots**: Individual PR curves for each class
- **Comprehensive Reports**: Detailed text reports with all metrics
- **TensorBoard Dashboards**: Real-time monitoring of all metrics

## Key Changes Made

### 1. Data Loading (`utils/data_loading.py`)
#### Modified `BasicDataset` class:
- Added `target_size` parameter (default: (640, 640))
- Modified `preprocess()` method to accept `target_size` parameter
- Updated `__getitem__()` to:
  - Store original image size as `original_size`
  - Include image name for debugging
  - Use `target_size` for resizing during training

### 2. Training (`train.py`)
#### New Functions Added:
- `calculate_class_wise_metrics()`: Computes class-wise mAP@IoU0.5 and AP
- `plot_pr_curves()`: Creates PR curve plots for all classes

#### Modified `train_model()` function:
- Added best model tracking variables
- Enhanced validation loops with class-wise metric calculation
- Added PR curve generation at each validation step
- Implemented best model saving logic
- Added comprehensive end-of-training reporting

#### Enhanced Logging:
- Class-wise metrics logged to TensorBoard
- PR curves saved as images
- Best model summary generated
- Comprehensive training reports

### 3. Model Analysis (`analyze_best_model.py`)
#### New comprehensive analysis script:
- Loads best model and generates detailed performance report
- Creates individual PR curves for each class
- Calculates comprehensive metrics on validation set
- Generates detailed text reports

### 4. Enhanced Evaluation (`evaluate.py`)
- Modified to handle original size resizing
- Maintains accurate evaluation on original image dimensions

## Usage Examples

### 1. Training with PR Curves and Best Model Tracking
```bash
# Train with comprehensive metrics tracking
python train.py --epochs 50 --batch-size 4 --learning-rate 1e-4 --amp --classes 2

# Train with custom target size
python train.py --epochs 50 --target-width 512 --target-height 512 --classes 3
```

### 2. Using the Enhanced Training Script
```bash
python train_640x640.py
```

### 3. Analyzing the Best Model
```bash
# Comprehensive analysis of the best model
python analyze_best_model.py --model checkpoints/best_model.pth --val-data path/to/validation/data

# With custom parameters
python analyze_best_model.py --model checkpoints/best_model.pth --val-data path/to/val --batch-size 8 --target-width 640 --target-height 640
```

## Output Files Generated

### During Training:
- `checkpoints/best_model.pth`: Best model based on validation loss
- `checkpoints/best_pr_curves.png`: PR curves from the best epoch
- `checkpoints/pr_curves_epoch_X.png`: PR curves for each epoch
- `checkpoints/best_model_summary.txt`: Detailed best model report
- TensorBoard logs with all metrics and PR curves

### Analysis Output:
- `checkpoints/analysis/comprehensive_pr_curves.png`: Overall PR curve plot
- `checkpoints/analysis/pr_curve_class_X.png`: Individual class PR curves
- `checkpoints/analysis/model_analysis_report.txt`: Comprehensive performance report

## Metrics Tracked

### Class-wise Metrics:
- **Average Precision (AP)**: Area under the PR curve
- **mAP@IoU0.5**: Mean Average Precision at IoU threshold 0.5
- **Precision-Recall Curves**: Full curves for visualization
- **Number of Predictions**: Sample count per class

### Overall Metrics:
- **Overall mAP@IoU0.5**: Average across all classes
- **Validation Loss**: Primary metric for best model selection
- **Dice Score**: Additional segmentation metric
- **Comprehensive Metrics**: IoU, F1, Accuracy, etc.

## TensorBoard Visualization

Access comprehensive real-time training monitoring:
```bash
tensorboard --logdir=runs
```

**Available Dashboards:**
- **Scalars**: All metrics including class-wise mAP and AP
- **Images**: PR curve plots and training visualizations  
- **PR Curves**: Interactive precision-recall curves
- **Histograms**: Model weights and gradients

## Best Model Information

The best model tracking provides:
```
Best Model Summary:
- Epoch: 34
- Validation Loss: 0.234567
- Overall mAP@IoU0.5: 0.8234
- Class-wise Performance:
  - class_0: AP=0.8123, mAP@IoU0.5=0.7890
  - class_1: AP=0.8345, mAP@IoU0.5=0.8578
```

## Performance Benefits

### Enhanced Training Insights:
- **Class Performance**: Understand per-class strengths and weaknesses
- **Training Progress**: Monitor mAP improvements over epochs
- **Best Model Selection**: Automatic selection based on validation performance
- **Comprehensive Analysis**: Detailed post-training analysis capabilities

### Research and Development:
- **Reproducible Results**: Best model state fully preserved
- **Detailed Metrics**: All evaluation metrics for thorough analysis
- **Visual Analysis**: PR curves for easy interpretation
- **Comparative Studies**: Easy comparison between different training runs

## Memory and Performance Considerations

### Training:
- **GPU Memory**: 640x640 images require ~4-8GB for batch_size=4
- **Storage**: PR curve images and comprehensive logs require additional disk space
- **Computation**: Class-wise metric calculation adds ~5-10% overhead

### Recommendations:
- Use `--amp` for mixed precision training
- Adjust batch size based on available GPU memory
- Monitor disk space for comprehensive logging
- Use TensorBoard for real-time monitoring

This enhanced training pipeline provides production-ready model training with comprehensive evaluation, making it suitable for research, development, and deployment scenarios.
