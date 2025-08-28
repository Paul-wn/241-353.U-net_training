#!/usr/bin/env python3
"""
Example script demonstrating how to train UNet with 640x640 target size,
automatic resizing back to original dimensions for evaluation,
PR curve plotting with mAP @IoU0.5 for each class, and best model saving.
"""

import logging
import torch
from unet import UNet
from train import train_model

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    # Model configuration
    model = UNet(n_channels=3, n_classes=2, bilinear=False)  # Change n_classes as needed
    model = model.to(memory_format=torch.channels_last)
    model.to(device=device)
    
    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')
    
    # Training configuration with 640x640 target size
    target_size = (640, 640)  # (width, height)
    
    logging.info(f"Training Features Enabled:")
    logging.info(f"✓ Fixed 640x640 input size for training stability")
    logging.info(f"✓ Automatic resizing back to original dimensions for evaluation")
    logging.info(f"✓ Class-wise mAP@IoU0.5 calculation and logging")
    logging.info(f"✓ PR curve plotting for each class")
    logging.info(f"✓ Best model saving based on lowest validation loss")
    logging.info(f"✓ Comprehensive metrics reporting")
    
    try:
        train_model(
            model=model,
            epochs=50,                    # Number of epochs
            batch_size=4,                 # Batch size (adjust based on GPU memory)
            learning_rate=1e-4,           # Learning rate
            device=device,
            img_scale=1.0,                # Keep at 1.0 since we're using target_size
            amp=True,                     # Use mixed precision for faster training
            weight_decay=1e-8,
            momentum=0.999,
            gradient_clipping=1.0,
            target_size=target_size       # Fixed 640x640 for training
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider reducing batch size or enabling AMP.')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        
        # Retry with checkpointing and reduced batch size
        train_model(
            model=model,
            epochs=50,
            batch_size=2,                 # Reduced batch size
            learning_rate=1e-4,
            device=device,
            img_scale=1.0,
            amp=True,
            weight_decay=1e-8,
            momentum=0.999,
            gradient_clipping=1.0,
            target_size=target_size
        )
    
    logging.info('Training completed!')
    logging.info("\nTo analyze the best model, run:")
    logging.info("python analyze_best_model.py --model checkpoints/best_model.pth --val-data path/to/validation/data")

if __name__ == '__main__':
    main()
