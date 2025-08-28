#!/usr/bin/env python3
"""
Script to analyze the best trained model and generate comprehensive PR curves and mAP reports.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from torch.utils.data import DataLoader
from train import calculate_class_wise_metrics, plot_pr_curves

def load_best_model(checkpoint_path, device):
    """Load the best model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Determine model parameters from checkpoint
    model_state = checkpoint['model_state_dict']
    
    # Infer number of classes from the final layer
    n_classes = model_state['outc.conv.weight'].shape[0]
    n_channels = model_state['inc.double_conv.0.weight'].shape[1]
    
    # Create model
    model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=False)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    
    return model, checkpoint

def analyze_best_model(model_path, val_data_dir, target_size=(640, 640), batch_size=4):
    """
    Analyze the best model and generate comprehensive reports
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Load best model
    model_path = Path(model_path)
    model, checkpoint = load_best_model(model_path, device)
    
    logging.info(f"Loaded best model from epoch {checkpoint['epoch']}")
    logging.info(f"Best validation loss: {checkpoint['val_loss']:.6f}")
    logging.info(f"Overall mAP@IoU0.5: {checkpoint['overall_map']:.4f}")
    
    # Load validation dataset
    val_img_dir = Path(val_data_dir) / 'images'
    val_mask_dir = Path(val_data_dir) / 'masks'
    
    try:
        val_dataset = CarvanaDataset(val_img_dir, val_mask_dir, scale=1.0, target_size=target_size)
    except (AssertionError, RuntimeError, IndexError):
        val_dataset = BasicDataset(val_img_dir, val_mask_dir, scale=1.0, 
                                 mask_suffix='_colored_mask_bright', target_size=target_size)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=0, pin_memory=True)
    
    logging.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Collect all predictions
    all_predictions = []
    all_true_masks = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Generating predictions"):
            images, true_masks = batch['image'], batch['mask']
            original_sizes = batch.get('original_size', None)
            
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = true_masks.to(device=device, dtype=torch.long)
            
            predictions = model(images)
            
            # Resize predictions back to original size if available
            if original_sizes is not None:
                for i in range(len(predictions)):
                    orig_size = original_sizes[i]
                    pred_resized = F.interpolate(
                        predictions[i:i+1], 
                        size=(orig_size[1], orig_size[0]),
                        mode='bilinear'
                    )
                    true_resized = F.interpolate(
                        true_masks[i:i+1].float().unsqueeze(1), 
                        size=(orig_size[1], orig_size[0]),
                        mode='nearest'
                    ).squeeze(1).long()
                    
                    all_predictions.append(pred_resized.squeeze(0))
                    all_true_masks.append(true_resized.squeeze(0))
            else:
                for i in range(len(predictions)):
                    all_predictions.append(predictions[i])
                    all_true_masks.append(true_masks[i])
    
    # Calculate comprehensive metrics
    class_metrics, pr_curves, overall_map = calculate_class_wise_metrics(
        all_predictions, all_true_masks, model.n_classes
    )
    
    # Create output directory
    output_dir = model_path.parent / 'analysis'
    output_dir.mkdir(exist_ok=True)
    
    # Generate comprehensive PR curve plot
    pr_plot_path = output_dir / 'comprehensive_pr_curves.png'
    plot_pr_curves(pr_curves, pr_plot_path, checkpoint['epoch'], overall_map)
    
    # Generate detailed report
    report_path = output_dir / 'model_analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE MODEL ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Model Information:\n")
        f.write(f"- Best epoch: {checkpoint['epoch']}\n")
        f.write(f"- Validation loss: {checkpoint['val_loss']:.6f}\n")
        f.write(f"- Number of classes: {model.n_classes}\n")
        f.write(f"- Target training size: {target_size}\n")
        f.write(f"- Validation samples: {len(all_predictions)}\n\n")
        
        f.write(f"Overall Performance:\n")
        f.write(f"- Overall mAP@IoU0.5: {overall_map:.4f}\n\n")
        
        f.write("Class-wise Performance:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Class':<15} {'AP':<10} {'mAP@IoU0.5':<12} {'Samples':<10}\n")
        f.write("-" * 60 + "\n")
        
        for class_name, metrics in class_metrics.items():
            f.write(f"{class_name:<15} {metrics['ap']:<10.4f} {metrics['map_iou05']:<12.4f} {metrics['num_predictions']:<10}\n")
        
        f.write("\nDetailed Class Metrics:\n")
        f.write("="*50 + "\n")
        for class_name, metrics in class_metrics.items():
            f.write(f"\n{class_name.upper()}:\n")
            f.write(f"  Average Precision (AP): {metrics['ap']:.6f}\n")
            f.write(f"  mAP@IoU0.5: {metrics['map_iou05']:.6f}\n")
            f.write(f"  Number of predictions: {metrics['num_predictions']}\n")
            f.write(f"  Precision curve points: {len(metrics['precision'])}\n")
            f.write(f"  Recall curve points: {len(metrics['recall'])}\n")
    
    # Generate individual class PR curves
    for class_name, curve_data in pr_curves.items():
        class_plot_path = output_dir / f'pr_curve_{class_name}.png'
        
        plt.figure(figsize=(10, 8))
        plt.plot(curve_data['recall'], curve_data['precision'], 'b-', linewidth=2)
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {class_name}\nAP = {curve_data["ap"]:.4f}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.savefig(class_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Print summary
    print("\n" + "="*80)
    print("MODEL ANALYSIS COMPLETE")
    print("="*80)
    print(f"Overall mAP@IoU0.5: {overall_map:.4f}")
    print("\nClass-wise Performance:")
    for class_name, metrics in class_metrics.items():
        print(f"  {class_name}: AP={metrics['ap']:.4f}, mAP@IoU0.5={metrics['map_iou05']:.4f}")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"- Comprehensive report: {report_path}")
    print(f"- PR curves plot: {pr_plot_path}")
    print(f"- Individual class plots: {output_dir}/pr_curve_class_*.png")
    
    return class_metrics, pr_curves, overall_map

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze the best trained UNet model')
    parser.add_argument('--model', '-m', required=True, help='Path to best_model.pth')
    parser.add_argument('--val-data', '-v', required=True, help='Path to validation data directory')
    parser.add_argument('--target-width', type=int, default=640, help='Target width used during training')
    parser.add_argument('--target-height', type=int, default=640, help='Target height used during training')
    parser.add_argument('--batch-size', '-b', type=int, default=4, help='Batch size for analysis')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    target_size = (args.target_width, args.target_height)
    
    analyze_best_model(
        model_path=args.model,
        val_data_dir=args.val_data,
        target_size=target_size,
        batch_size=args.batch_size
    )
