import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

dir_img = Path('../Dataset/U-net_Dataset/train/images/')
dir_mask = Path('../Dataset/U-net_Dataset/train/masks/')
dir_val_img = Path('../Dataset/U-net_Dataset/val/images/')
dir_val_mask = Path('../Dataset/U-net_Dataset/val/masks/')
dir_checkpoint = Path('./checkpoints/')

try: 
    from torch.utils.tensorboard import SummaryWriter
    tensorboard_available = True
except:
    print("Tensorboard not available. Install with: pip install tensorboard")
    tensorboard_available = False

def calculate_class_wise_metrics(predictions, true_masks, num_classes, iou_threshold=0.5):
    """
    Calculate class-wise mAP@IoU0.5 and generate PR curves
    
    Args:
        predictions: List of prediction tensors
        true_masks: List of true mask tensors  
        num_classes: Number of classes
        iou_threshold: IoU threshold for mAP calculation
    
    Returns:
        Dictionary with class-wise metrics and overall mAP
    """
    class_metrics = {}
    class_aps = []
    pr_curves = {}
    
    for class_idx in range(num_classes):
        # Extract binary masks for current class
        class_probs = []
        class_labels = []
        class_ious = []
        
        for pred, true in zip(predictions, true_masks):
            if num_classes == 1:
                # Binary segmentation
                pred_prob = torch.sigmoid(pred).squeeze()
                pred_binary = (pred_prob > 0.5).float()
                true_binary = true.float()
            else:
                # Multi-class segmentation
                pred_prob = F.softmax(pred, dim=1)[class_idx]
                pred_binary = (torch.argmax(F.softmax(pred, dim=1), dim=1) == class_idx).float()
                true_binary = (true == class_idx).float()
            
            # Calculate IoU for this prediction
            intersection = (pred_binary * true_binary).sum()
            union = pred_binary.sum() + true_binary.sum() - intersection
            iou = intersection / union if union > 0 else 0.0
            
            # Store probabilities and labels for PR curve
            class_probs.append(pred_prob.flatten().cpu().numpy())
            class_labels.append(true_binary.flatten().cpu().numpy())
            class_ious.append(iou.item())
        
        # Concatenate all predictions for this class
        all_probs = np.concatenate(class_probs)
        all_labels = np.concatenate(class_labels)
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
        
        # Calculate Average Precision
        ap = average_precision_score(all_labels, all_probs)
        
        # Calculate mAP@IoU0.5
        map_iou05 = sum(1 for iou in class_ious if iou >= iou_threshold) / len(class_ious) if class_ious else 0.0
        
        class_metrics[f'class_{class_idx}'] = {
            'ap': ap,
            'map_iou05': map_iou05,
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'num_predictions': len(class_ious)
        }
        
        pr_curves[f'class_{class_idx}'] = {
            'precision': precision,
            'recall': recall,
            'ap': ap
        }
        
        class_aps.append(ap)
    
    # Calculate overall mAP
    overall_map = np.mean(class_aps) if class_aps else 0.0
    
    return class_metrics, pr_curves, overall_map

def plot_pr_curves(pr_curves, save_path, epoch, overall_map):
    """
    Plot PR curves for all classes
    """
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, (class_name, curve_data) in enumerate(pr_curves.items()):
        color = colors[i % len(colors)]
        plt.plot(curve_data['recall'], curve_data['precision'], 
                color=color, linewidth=2, 
                label=f'{class_name} (AP = {curve_data["ap"]:.3f})')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curves - Epoch {epoch}\nOverall mAP@IoU0.5 = {overall_map:.3f}', fontsize=14)
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def calculate_comprehensive_metrics(all_probs, all_labels, all_bin_preds, all_true_masks):
    """
    Calculate comprehensive metrics similar to YOLOv5
    """
    metrics = {}
    
    # Convert to numpy
    probs_np = torch.cat(all_probs).detach().cpu().numpy()
    labels_np = torch.cat(all_labels).detach().cpu().numpy()
    
    # Flatten binary predictions and true masks for pixel-wise metrics
    bin_preds_flat = torch.cat([pred.flatten() for pred in all_bin_preds]).cpu().numpy()
    true_masks_flat = torch.cat([mask.flatten() for mask in all_true_masks]).cpu().numpy()
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(labels_np, (probs_np > 0.5).astype(int)).ravel()
    
    # Basic metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Average Precision (AP)
    try:
        ap = average_precision_score(labels_np, probs_np)
    except:
        ap = 0.0
    
    # IoU calculation
    intersection = np.logical_and(bin_preds_flat, true_masks_flat).sum()
    union = np.logical_or(bin_preds_flat, true_masks_flat).sum()
    iou = intersection / union if union > 0 else 0.0
    
    # Dice coefficient
    dice = (2.0 * intersection) / (bin_preds_flat.sum() + true_masks_flat.sum()) if (bin_preds_flat.sum() + true_masks_flat.sum()) > 0 else 0.0
    
    # Calculate IoU for each individual prediction (proper mAP calculation)
    individual_ious = []
    
    # Calculate IoU per image/batch
    for i in range(len(all_bin_preds)):
        pred_flat = all_bin_preds[i].flatten().numpy()
        true_flat = all_true_masks[i].flatten().numpy()
        
        intersection = np.logical_and(pred_flat, true_flat).sum()
        union = np.logical_or(pred_flat, true_flat).sum()
        iou_individual = intersection / union if union > 0 else 0.0
        individual_ious.append(iou_individual)
    
    # mAP calculation at different IoU thresholds
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps = []
    for thresh in iou_thresholds:
        # Count how many predictions meet this IoU threshold
        correct_predictions = sum(1 for iou_val in individual_ious if iou_val >= thresh)
        ap_thresh = correct_predictions / len(individual_ious) if len(individual_ious) > 0 else 0.0
        aps.append(ap_thresh)
    
    map_50_95 = np.mean(aps)
    map_50 = sum(1 for iou_val in individual_ious if iou_val >= 0.5) / len(individual_ious) if len(individual_ious) > 0 else 0.0
    map_75 = sum(1 for iou_val in individual_ious if iou_val >= 0.75) / len(individual_ious) if len(individual_ious) > 0 else 0.0
    
    metrics.update({
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'specificity': specificity,
        'iou': iou,
        'dice': dice,
        'ap': ap,
        'map_50': map_50,
        'map_75': map_75,
        'map_50_95': map_50_95,
        'mean_individual_iou': np.mean(individual_ious) if individual_ious else 0.0,
        'individual_ious_above_50': sum(1 for x in individual_ious if x >= 0.5),
        'total_predictions': len(individual_ious),
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    })
    
    # Debug information
    if len(individual_ious) > 0:
        print(f"Debug mAP calculation:")
        print(f"  Number of predictions: {len(individual_ious)}")
        print(f"  Individual IoUs (first 5): {individual_ious[:5]}")
        print(f"  Mean IoU: {np.mean(individual_ious):.4f}")
        print(f"  IoUs >= 0.5: {sum(1 for x in individual_ious if x >= 0.5)} / {len(individual_ious)}")
        print(f"  IoUs >= 0.75: {sum(1 for x in individual_ious if x >= 0.75)} / {len(individual_ious)}")
        print(f"  mAP@0.5: {map_50:.4f}")
        print(f"  mAP@0.75: {map_75:.4f}")
        print(f"  mAP@0.5:0.95: {map_50_95:.4f}")
        print(f"  Overall IoU (pixel-wise): {iou:.4f}")
        print(f"  Intersection: {intersection}, Union: {union}")
        print(f"  Positive predictions: {bin_preds_flat.sum()}, True positives: {true_masks_flat.sum()}")
    
    return metrics

def log_metrics_to_tensorboard(writer, metrics, global_step, prefix='Val'):
    """Log comprehensive metrics to TensorBoard"""
    if not writer:
        return
        
    # Main metrics
    writer.add_scalar(f'{prefix}/Precision', metrics['precision'], global_step)
    writer.add_scalar(f'{prefix}/Recall', metrics['recall'], global_step)
    writer.add_scalar(f'{prefix}/F1_Score', metrics['f1_score'], global_step)
    writer.add_scalar(f'{prefix}/Accuracy', metrics['accuracy'], global_step)
    writer.add_scalar(f'{prefix}/Specificity', metrics['specificity'], global_step)
    writer.add_scalar(f'{prefix}/IoU', metrics['iou'], global_step)
    writer.add_scalar(f'{prefix}/Dice', metrics['dice'], global_step)
    writer.add_scalar(f'{prefix}/AP', metrics['ap'], global_step)
    
    # mAP metrics
    writer.add_scalar(f'{prefix}/mAP_50', metrics['map_50'], global_step)
    writer.add_scalar(f'{prefix}/mAP_75', metrics['map_75'], global_step)
    writer.add_scalar(f'{prefix}/mAP_50_95', metrics['map_50_95'], global_step)
    
    # Confusion matrix components
    writer.add_scalar(f'{prefix}/True_Positives', metrics['true_positives'], global_step)
    writer.add_scalar(f'{prefix}/False_Positives', metrics['false_positives'], global_step)
    writer.add_scalar(f'{prefix}/True_Negatives', metrics['true_negatives'], global_step)
    writer.add_scalar(f'{prefix}/False_Negatives', metrics['false_negatives'], global_step)
    
    # Additional debugging metrics
    if 'mean_individual_iou' in metrics:
        writer.add_scalar(f'{prefix}/Mean_Individual_IoU', metrics['mean_individual_iou'], global_step)
        writer.add_scalar(f'{prefix}/Predictions_Above_IoU_50', metrics['individual_ious_above_50'], global_step)
        writer.add_scalar(f'{prefix}/Total_Predictions', metrics['total_predictions'], global_step)

def create_metrics_summary_plot(metrics, save_path):
    """Create a comprehensive metrics summary plot similar to YOLOv5"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Main metrics bar chart
    main_metrics = ['precision', 'recall', 'f1_score', 'accuracy', 'iou', 'dice']
    main_values = [metrics[m] for m in main_metrics]
    main_labels = ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'IoU', 'Dice']
    
    bars1 = ax1.bar(main_labels, main_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    ax1.set_title('Main Performance Metrics', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, main_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: mAP metrics
    map_metrics = ['map_50', 'map_75', 'map_50_95', 'ap']
    map_values = [metrics[m] for m in map_metrics]
    map_labels = ['mAP@0.5', 'mAP@0.75', 'mAP@0.5:0.95', 'AP']
    
    bars2 = ax2.bar(map_labels, map_values, color=['#17becf', '#bcbd22', '#ff9896', '#c5b0d5'])
    ax2.set_title('Average Precision Metrics', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars2, map_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Confusion Matrix visualization
    cm_data = np.array([[metrics['true_negatives'], metrics['false_positives']],
                        [metrics['false_negatives'], metrics['true_positives']]])
    im = ax3.imshow(cm_data, interpolation='nearest', cmap='Blues')
    ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    
    # Add text annotations
    thresh = cm_data.max() / 2.
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, format(cm_data[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm_data[i, j] > thresh else "black",
                    fontsize=12, fontweight='bold')
    
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(['Negative', 'Positive'])
    ax3.set_yticklabels(['Negative', 'Positive'])
    
    # Plot 4: Model summary statistics
    ax4.axis('off')
    summary_text = f"""
Model Performance Summary

📊 Overall Metrics:
• Accuracy: {metrics['accuracy']:.4f}
• Precision: {metrics['precision']:.4f}
• Recall: {metrics['recall']:.4f}
• F1-Score: {metrics['f1_score']:.4f}

🎯 Segmentation Metrics:
• IoU: {metrics['iou']:.4f}
• Dice: {metrics['dice']:.4f}
• AP: {metrics['ap']:.4f}

📈 mAP Metrics:
• mAP@0.5: {metrics['map_50']:.4f}
• mAP@0.75: {metrics['map_75']:.4f}
• mAP@0.5:0.95: {metrics['map_50_95']:.4f}

🔢 Confusion Matrix:
• TP: {metrics['true_positives']:<8} FP: {metrics['false_positives']}
• FN: {metrics['false_negatives']:<8} TN: {metrics['true_negatives']}
"""
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def print_final_metrics(metrics, dataset_name):
    """Print comprehensive metrics in YOLOv5 style"""
    print(f"\n{'='*60}")
    print(f"{dataset_name} Results")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Value':<15} {'Description'}")
    print(f"{'-'*60}")
    print(f"{'Precision':<20} {metrics['precision']:<15.4f} TP/(TP+FP)")
    print(f"{'Recall':<20} {metrics['recall']:<15.4f} TP/(TP+FN)")
    print(f"{'F1-Score':<20} {metrics['f1_score']:<15.4f} 2*P*R/(P+R)")
    print(f"{'Accuracy':<20} {metrics['accuracy']:<15.4f} (TP+TN)/(TP+TN+FP+FN)")
    print(f"{'Specificity':<20} {metrics['specificity']:<15.4f} TN/(TN+FP)")
    print(f"{'IoU':<20} {metrics['iou']:<15.4f} Intersection over Union")
    print(f"{'Dice':<20} {metrics['dice']:<15.4f} Dice Coefficient")
    print(f"{'AP':<20} {metrics['ap']:<15.4f} Average Precision")
    print(f"{'mAP@0.5':<20} {metrics['map_50']:<15.4f} mAP at IoU=0.5")
    print(f"{'mAP@0.75':<20} {metrics['map_75']:<15.4f} mAP at IoU=0.75")
    print(f"{'mAP@0.5:0.95':<20} {metrics['map_50_95']:<15.4f} mAP at IoU=0.5:0.95")
    print(f"{'-'*60}")
    print(f"{'Confusion Matrix:'}")
    print(f"{'TP:':<8} {metrics['true_positives']:<12} {'FP:':<8} {metrics['false_positives']}")
    print(f"{'FN:':<8} {metrics['false_negatives']:<12} {'TN:':<8} {metrics['true_negatives']}")
    print(f"{'='*60}\n")

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        target_size: tuple = (640, 640),
):
    # 1. Create datasets with fixed target size for training
    try:
        train_dataset = CarvanaDataset(dir_img, dir_mask, img_scale, target_size=target_size)
    except (AssertionError, RuntimeError, IndexError):
        train_dataset = BasicDataset(dir_img, dir_mask, img_scale, mask_suffix='_colored_mask_bright', target_size=target_size)
    
    try:
        val_dataset = CarvanaDataset(dir_val_img, dir_val_mask, img_scale, target_size=target_size)
    except (AssertionError, RuntimeError, IndexError):
        val_dataset = BasicDataset(dir_val_img, dir_val_mask, img_scale, mask_suffix='_colored_mask_bright', target_size=target_size)

    # 2. Get dataset sizes
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp, target_size=target_size)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Target size:     {target_size}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.amp.GradScaler(device.type, enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # Initialize TensorBoard writer
    writer = SummaryWriter() if tensorboard_available else None
    
    # Best model tracking
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                # Log training loss to TensorBoard
                if writer:
                    writer.add_scalar('Train/Loss', loss.item(), global_step)

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        # Compute validation loss and comprehensive metrics for TensorBoard
                        if writer:
                            model.eval()
                            val_loss = 0.0
                            all_probs, all_labels, all_ious = [], [], []
                            all_bin_preds, all_true_masks = [], []
                            with torch.no_grad():
                                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                                    for val_batch in val_loader:
                                        val_images, val_true_masks = val_batch['image'], val_batch['mask']
                                        original_sizes = val_batch.get('original_size', None)
                                        
                                        val_images = val_images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                                        val_true_masks = val_true_masks.to(device=device, dtype=torch.long)
                                        val_masks_pred = model(val_images)
                                        
                                        # Resize predictions back to original size if available
                                        if original_sizes is not None:
                                            resized_preds = []
                                            resized_trues = []
                                            
                                            for i in range(len(val_masks_pred)):
                                                orig_size = original_sizes[i]
                                                # Resize prediction back to original size
                                                pred_resized = F.interpolate(
                                                    val_masks_pred[i:i+1], 
                                                    size=(orig_size[1], orig_size[0]),  # (height, width)
                                                    mode='bilinear'
                                                )
                                                # Resize true mask back to original size  
                                                true_resized = F.interpolate(
                                                    val_true_masks[i:i+1].float().unsqueeze(1), 
                                                    size=(orig_size[1], orig_size[0]),  # (height, width)
                                                    mode='nearest'
                                                ).squeeze(1).long()
                                                
                                                resized_preds.append(pred_resized)
                                                resized_trues.append(true_resized)
                                            
                                            val_masks_pred_eval = torch.cat(resized_preds, dim=0)
                                            val_true_masks_eval = torch.cat(resized_trues, dim=0)
                                        else:
                                            val_masks_pred_eval = val_masks_pred
                                            val_true_masks_eval = val_true_masks
                                        
                                        # compute loss (use original 640x640 for loss calculation)
                                        if model.n_classes == 1:
                                            loss_val = criterion(val_masks_pred.squeeze(1), val_true_masks.float())
                                            loss_val += dice_loss(F.sigmoid(val_masks_pred.squeeze(1)), val_true_masks.float(), multiclass=False)
                                            # Use resized predictions for metrics
                                            probs = torch.sigmoid(val_masks_pred_eval).contiguous().view(-1)
                                            labels = val_true_masks_eval.float().contiguous().view(-1)
                                            bin_preds = (torch.sigmoid(val_masks_pred_eval) > 0.5).squeeze(1)
                                        else:
                                            loss_val = criterion(val_masks_pred, val_true_masks)
                                            loss_val += dice_loss(
                                                F.softmax(val_masks_pred, dim=1).float(),
                                                F.one_hot(val_true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                                                multiclass=True
                                            )
                                            # For multi-class: use argmax to get predicted class (using resized predictions for metrics)
                                            pred_classes = torch.argmax(F.softmax(val_masks_pred_eval, dim=1), dim=1)
                                            # Create binary mask: non-background (class > 0) vs background (class == 0)
                                            probs = (pred_classes > 0).float().contiguous().view(-1)
                                            labels = (val_true_masks_eval > 0).float().contiguous().view(-1)
                                            bin_preds = (pred_classes > 0)
                                        
                                        val_loss += loss_val.item()
                                        all_probs.append(probs.cpu())
                                        all_labels.append(labels.cpu())
                                        all_bin_preds.append(bin_preds.cpu())
                                        all_true_masks.append((val_true_masks_eval > 0).cpu())
                                        
                                        # compute IoU per image for backward compatibility (using resized predictions)
                                        for pred_mask, true_mask in zip(bin_preds, val_true_masks_eval):
                                            inter = (pred_mask & (true_mask > 0)).sum().float()
                                            uni = (pred_mask | (true_mask > 0)).sum().float()
                                            all_ious.append((inter / uni).item() if uni > 0 else 0.0)
                            
                            # Calculate comprehensive metrics
                            metrics = calculate_comprehensive_metrics(all_probs, all_labels, all_bin_preds, all_true_masks)
                            
                            # Calculate class-wise metrics and PR curves
                            val_predictions = []
                            val_true_masks_for_pr = []
                            
                            # Collect predictions for PR curve calculation
                            model.eval()
                            with torch.no_grad():
                                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                                    for val_batch in val_loader:
                                        val_images, val_true_masks = val_batch['image'], val_batch['mask']
                                        original_sizes = val_batch.get('original_size', None)
                                        
                                        val_images = val_images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                                        val_true_masks = val_true_masks.to(device=device, dtype=torch.long)
                                        val_masks_pred = model(val_images)
                                        
                                        # Resize predictions back to original size if available
                                        if original_sizes is not None:
                                            for i in range(len(val_masks_pred)):
                                                orig_size = original_sizes[i]
                                                pred_resized = F.interpolate(
                                                    val_masks_pred[i:i+1], 
                                                    size=(orig_size[1], orig_size[0]),
                                                    mode='bilinear'
                                                )
                                                true_resized = F.interpolate(
                                                    val_true_masks[i:i+1].float().unsqueeze(1), 
                                                    size=(orig_size[1], orig_size[0]),
                                                    mode='nearest'
                                                ).squeeze(1).long()
                                                
                                                val_predictions.append(pred_resized.squeeze(0))
                                                val_true_masks_for_pr.append(true_resized.squeeze(0))
                                        else:
                                            for i in range(len(val_masks_pred)):
                                                val_predictions.append(val_masks_pred[i])
                                                val_true_masks_for_pr.append(val_true_masks[i])
                            
                            # Calculate class-wise mAP and PR curves
                            class_metrics, pr_curves, overall_map = calculate_class_wise_metrics(
                                val_predictions, val_true_masks_for_pr, model.n_classes
                            )
                            
                            # Plot and save PR curves
                            pr_curve_path = dir_checkpoint / f'pr_curves_epoch_{epoch}_step_{global_step}.png'
                            dir_checkpoint.mkdir(parents=True, exist_ok=True)
                            plot_pr_curves(pr_curves, pr_curve_path, epoch, overall_map)
                            
                            # Log class-wise metrics to TensorBoard
                            for class_name, class_data in class_metrics.items():
                                writer.add_scalar(f'Val/AP_{class_name}', class_data['ap'], global_step)
                                writer.add_scalar(f'Val/mAP_IoU05_{class_name}', class_data['map_iou05'], global_step)
                                
                                # Add PR curve to TensorBoard
                                writer.add_pr_curve(f'Val/PR_{class_name}', 
                                                  np.ones_like(class_data['precision'][:-1]) if len(class_data['precision']) > 1 else np.array([1]), 
                                                  class_data['precision'][:-1] if len(class_data['precision']) > 1 else class_data['precision'], 
                                                  global_step)
                            
                            # Log overall mAP
                            writer.add_scalar('Val/Overall_mAP_IoU05', overall_map, global_step)
                            
                            num_batches = len(val_loader)
                            val_loss /= num_batches
                            
                            # Log to TensorBoard
                            writer.add_scalar('Val/Loss', val_loss, global_step)
                            log_metrics_to_tensorboard(writer, metrics, global_step, 'Val')
                            
                            # Log PR curve
                            probs_cat = torch.cat(all_probs)
                            labels_cat = torch.cat(all_labels)
                            writer.add_pr_curve('Val/PR', labels_cat.detach().numpy(), probs_cat.detach().numpy(), global_step)
                            
                            # Backward compatibility mAP calculation
                            mAP50_legacy = sum(i >= 0.5 for i in all_ious) / len(all_ious) if len(all_ious) > 0 else 0.0
                            writer.add_scalar('Val/mAP_IoU50_Legacy', mAP50_legacy, global_step)
                            
                            # Check if this is the best model so far
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                best_epoch = epoch
                                best_model_state = {
                                    'epoch': epoch,
                                    'model_state_dict': model.state_dict().copy(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'val_loss': val_loss,
                                    'val_dice': val_score,
                                    'overall_map': overall_map,
                                    'class_metrics': class_metrics,
                                    'mask_values': train_dataset.mask_values
                                }
                                
                                # Save best model
                                best_model_path = dir_checkpoint / 'best_model.pth'
                                torch.save(best_model_state, best_model_path)
                                logging.info(f'New best model saved! Epoch {epoch}, Val Loss: {val_loss:.6f}, Overall mAP: {overall_map:.4f}')
                            
                            model.train()
                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        # Log validation metrics to TensorBoard
                        if writer:
                            writer.add_scalar('Val/Dice', val_score, global_step)
                            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], global_step)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            if model.n_classes == 1:
                                pred_mask_wandb = (torch.sigmoid(masks_pred) > 0.5).squeeze(1)[0].float().cpu()
                            else:
                                pred_mask_wandb = masks_pred.argmax(dim=1)[0].float().cpu()
                            
                            # Log to wandb with comprehensive metrics
                            wandb_log = {
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(pred_mask_wandb),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            }
                            
                            # Add comprehensive metrics to wandb if available
                            if writer and 'metrics' in locals():
                                wandb_log.update({
                                    'val_precision': metrics['precision'],
                                    'val_recall': metrics['recall'],
                                    'val_f1_score': metrics['f1_score'],
                                    'val_accuracy': metrics['accuracy'],
                                    'val_iou': metrics['iou'],
                                    'val_map_50': metrics['map_50'],
                                    'val_map_75': metrics['map_75'],
                                    'val_map_50_95': metrics['map_50_95']
                                })
                            
                            experiment.log(wandb_log)
                        except:
                            pass
        
        # Calculate validation loss at the end of each epoch
        model.eval()
        epoch_val_loss = 0.0
        epoch_predictions = []
        epoch_true_masks = []
        
        with torch.no_grad():
            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                for val_batch in val_loader:
                    val_images, val_true_masks = val_batch['image'], val_batch['mask']
                    original_sizes = val_batch.get('original_size', None)
                    
                    val_images = val_images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    val_true_masks = val_true_masks.to(device=device, dtype=torch.long)
                    val_masks_pred = model(val_images)
                    
                    # Resize predictions back to original size if available
                    if original_sizes is not None:
                        for i in range(len(val_masks_pred)):
                            orig_size = original_sizes[i]
                            pred_resized = F.interpolate(
                                val_masks_pred[i:i+1], 
                                size=(orig_size[1], orig_size[0]),
                                mode='bilinear'
                            )
                            true_resized = F.interpolate(
                                val_true_masks[i:i+1].float().unsqueeze(1), 
                                size=(orig_size[1], orig_size[0]),
                                mode='nearest'
                            ).squeeze(1).long()
                            
                            epoch_predictions.append(pred_resized.squeeze(0))
                            epoch_true_masks.append(true_resized.squeeze(0))
                    else:
                        for i in range(len(val_masks_pred)):
                            epoch_predictions.append(val_masks_pred[i])
                            epoch_true_masks.append(val_true_masks[i])
                    
                    # compute validation loss (use original size for loss)
                    if model.n_classes == 1:
                        loss_val = criterion(val_masks_pred.squeeze(1), val_true_masks.float())
                        loss_val += dice_loss(F.sigmoid(val_masks_pred.squeeze(1)), val_true_masks.float(), multiclass=False)
                    else:
                        loss_val = criterion(val_masks_pred, val_true_masks)
                        loss_val += dice_loss(
                            F.softmax(val_masks_pred, dim=1).float(),
                            F.one_hot(val_true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                    epoch_val_loss += loss_val.item()
        
        epoch_val_loss /= len(val_loader)
        
        # Calculate epoch-level class-wise metrics and PR curves
        epoch_class_metrics, epoch_pr_curves, epoch_overall_map = calculate_class_wise_metrics(
            epoch_predictions, epoch_true_masks, model.n_classes
        )
        
        # Plot epoch PR curves
        epoch_pr_path = dir_checkpoint / f'pr_curves_epoch_{epoch}.png'
        plot_pr_curves(epoch_pr_curves, epoch_pr_path, epoch, epoch_overall_map)
        
        # Check if this is the best model based on validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict().copy(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': epoch_val_loss,
                'overall_map': epoch_overall_map,
                'class_metrics': epoch_class_metrics,
                'mask_values': train_dataset.mask_values
            }
            
            # Save best model
            best_model_path = dir_checkpoint / 'best_model.pth'
            torch.save(best_model_state, best_model_path)
            
            # Save best PR curves
            best_pr_path = dir_checkpoint / 'best_pr_curves.png'
            plot_pr_curves(epoch_pr_curves, best_pr_path, epoch, epoch_overall_map)
            
            logging.info(f'New best model saved! Epoch {epoch}, Val Loss: {epoch_val_loss:.6f}, Overall mAP@IoU0.5: {epoch_overall_map:.4f}')
        
        model.train()
        
        # Log epoch-level metrics
        avg_epoch_loss = epoch_loss / len(train_loader)
        if writer:
            writer.add_scalar('Train/Epoch_Loss', avg_epoch_loss, epoch)
            writer.add_scalar('Val/Epoch_Loss', epoch_val_loss, epoch)
            writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('Val/Epoch_Overall_mAP_IoU05', epoch_overall_map, epoch)
            
            # Log class-wise epoch metrics
            for class_name, class_data in epoch_class_metrics.items():
                writer.add_scalar(f'Val/Epoch_AP_{class_name}', class_data['ap'], epoch)
                writer.add_scalar(f'Val/Epoch_mAP_IoU05_{class_name}', class_data['map_iou05'], epoch)
        
        logging.info(f'Epoch {epoch} completed. Train loss: {avg_epoch_loss:.6f}, Val loss: {epoch_val_loss:.6f}, Overall mAP@IoU0.5: {epoch_overall_map:.4f}')

        if save_checkpoint and epoch % 20 == 0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = train_dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint8b_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

    # Final comprehensive evaluation
    logging.info("Performing final comprehensive evaluation...")
    model.eval()
    final_val_loss = 0.0
    final_probs, final_labels = [], []
    final_bin_preds, final_true_masks = [], []
    
    with torch.no_grad():
        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
            for val_batch in tqdm(val_loader, desc="Final evaluation"):
                val_images, val_true_masks = val_batch['image'], val_batch['mask']
                original_sizes = val_batch.get('original_size', None)
                
                val_images = val_images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                val_true_masks = val_true_masks.to(device=device, dtype=torch.long)
                val_masks_pred = model(val_images)
                
                # Resize predictions back to original size if available
                if original_sizes is not None:
                    resized_preds = []
                    resized_trues = []
                    
                    for i in range(len(val_masks_pred)):
                        orig_size = original_sizes[i]
                        # Resize prediction back to original size
                        pred_resized = F.interpolate(
                            val_masks_pred[i:i+1], 
                            size=(orig_size[1], orig_size[0]),  # (height, width)
                            mode='bilinear'
                        )
                        # Resize true mask back to original size  
                        true_resized = F.interpolate(
                            val_true_masks[i:i+1].float().unsqueeze(1), 
                            size=(orig_size[1], orig_size[0]),  # (height, width)
                            mode='nearest'
                        ).squeeze(1).long()
                        
                        resized_preds.append(pred_resized)
                        resized_trues.append(true_resized)
                    
                    val_masks_pred_eval = torch.cat(resized_preds, dim=0)
                    val_true_masks_eval = torch.cat(resized_trues, dim=0)
                else:
                    val_masks_pred_eval = val_masks_pred
                    val_true_masks_eval = val_true_masks
                
                # compute loss (use original 640x640 for loss calculation)
                if model.n_classes == 1:
                    loss_val = criterion(val_masks_pred.squeeze(1), val_true_masks.float())
                    loss_val += dice_loss(F.sigmoid(val_masks_pred.squeeze(1)), val_true_masks.float(), multiclass=False)
                    # Use resized predictions for metrics
                    probs = torch.sigmoid(val_masks_pred_eval).contiguous().view(-1)
                    labels = val_true_masks_eval.float().contiguous().view(-1)
                    bin_preds = (torch.sigmoid(val_masks_pred_eval) > 0.5).squeeze(1)
                else:
                    loss_val = criterion(val_masks_pred, val_true_masks)
                    loss_val += dice_loss(
                        F.softmax(val_masks_pred, dim=1).float(),
                        F.one_hot(val_true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                        multiclass=True
                    )
                    # For multi-class: use argmax to get predicted class (using resized predictions for metrics)
                    pred_classes = torch.argmax(F.softmax(val_masks_pred_eval, dim=1), dim=1)
                    # Create binary mask: non-background (class > 0) vs background (class == 0)
                    probs = (pred_classes > 0).float().contiguous().view(-1)
                    labels = (val_true_masks_eval > 0).float().contiguous().view(-1)
                    bin_preds = (pred_classes > 0)
                
                final_val_loss += loss_val.item()
                final_probs.append(probs.cpu())
                final_labels.append(labels.cpu())
                final_bin_preds.append(bin_preds.cpu())
                final_true_masks.append((val_true_masks_eval > 0).cpu())
    
    # Calculate final comprehensive metrics
    final_metrics = calculate_comprehensive_metrics(final_probs, final_labels, final_bin_preds, final_true_masks)
    final_val_loss /= len(val_loader)
    
    # Print final results in YOLOv5 style
    print_final_metrics(final_metrics, "Final Validation")
    
    # Create and save comprehensive metrics summary plot
    plot_path = dir_checkpoint / 'final_metrics_summary.png'
    create_metrics_summary_plot(final_metrics, plot_path)
    logging.info(f"Metrics summary plot saved to {plot_path}")
    
    # Log final metrics to TensorBoard
    if writer:
        log_metrics_to_tensorboard(writer, final_metrics, global_step, 'Final')
        writer.add_scalar('Final/Loss', final_val_loss, global_step)
        
        # Add the summary plot to TensorBoard
        summary_image = plt.imread(plot_path)
        writer.add_image('Final/Metrics_Summary', summary_image, global_step, dataformats='HWC')
    
    # Save final metrics to file
    metrics_file = dir_checkpoint / 'final_metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write(f"Final Training Results\n")
        f.write(f"{'='*50}\n")
        f.write(f"Epochs trained: {epochs}\n")
        f.write(f"Final validation loss: {final_val_loss:.6f}\n")
        f.write(f"{'='*50}\n")
        for key, value in final_metrics.items():
            f.write(f"{key}: {value:.6f}\n")
    
    logging.info(f"Final metrics saved to {metrics_file}")
    
    # Report best model information
    if best_model_state is not None:
        logging.info(f"\n{'='*60}")
        logging.info(f"BEST MODEL SUMMARY")
        logging.info(f"{'='*60}")
        logging.info(f"Best model saved at epoch: {best_epoch}")
        logging.info(f"Best validation loss: {best_val_loss:.6f}")
        logging.info(f"Best overall mAP@IoU0.5: {best_model_state['overall_map']:.4f}")
        
        # Log class-wise performance of best model
        logging.info(f"\nClass-wise Performance (Best Model):")
        logging.info(f"{'-'*60}")
        for class_name, class_data in best_model_state['class_metrics'].items():
            logging.info(f"{class_name}: AP={class_data['ap']:.4f}, mAP@IoU0.5={class_data['map_iou05']:.4f}")
        
        # Save best model summary
        best_summary_file = dir_checkpoint / 'best_model_summary.txt'
        with open(best_summary_file, 'w') as f:
            f.write(f"Best Model Summary\n")
            f.write(f"{'='*50}\n")
            f.write(f"Epoch: {best_epoch}\n")
            f.write(f"Validation Loss: {best_val_loss:.6f}\n")
            f.write(f"Overall mAP@IoU0.5: {best_model_state['overall_map']:.4f}\n")
            f.write(f"{'='*50}\n")
            f.write(f"Class-wise Performance:\n")
            for class_name, class_data in best_model_state['class_metrics'].items():
                f.write(f"{class_name}:\n")
                f.write(f"  - Average Precision: {class_data['ap']:.6f}\n")
                f.write(f"  - mAP@IoU0.5: {class_data['map_iou05']:.6f}\n")
                f.write(f"  - Number of predictions: {class_data['num_predictions']}\n")
        
        logging.info(f"Best model summary saved to {best_summary_file}")
        logging.info(f"Best model weights saved to: {dir_checkpoint / 'best_model.pth'}")
        logging.info(f"Best PR curves saved to: {dir_checkpoint / 'best_pr_curves.png'}")
        logging.info(f"{'='*60}\n")
    else:
        logging.warning("No best model was saved during training!")

    # Close TensorBoard writer
    if writer:
        writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--target-width', type=int, default=640, help='Target width for training (default: 640)')
    parser.add_argument('--target-height', type=int, default=640, help='Target height for training (default: 640)')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    target_size = (args.target_width, args.target_height)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            amp=args.amp,
            target_size=target_size
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            amp=args.amp,
            target_size=target_size
        )
