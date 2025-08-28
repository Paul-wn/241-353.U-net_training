import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
import matplotlib.pyplot as plt

def sigmoid_heatmap(output, save_path=None):
    """
    Visualizes two heatmaps derived from sigmoid of model output.

    Parameters:
    - output (torch.Tensor): Model output tensor.
    - save_path (str, optional): Path to save the visualization image.

    """
    # Apply sigmoid to keep values between 0-1
    heatmap = torch.sigmoid(output).squeeze().cpu().numpy()
    heatmap0 = heatmap[0]
    heatmap1 = heatmap[1]

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Heatmap 0
    im0 = axs[0].imshow(heatmap0, cmap='hot', interpolation='nearest')
    axs[0].set_title('Prediction Heatmap (Background)')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04, label='Probability')

    # Heatmap 1
    im1 = axs[1].imshow(heatmap1, cmap='hot', interpolation='nearest')
    axs[1].set_title('Prediction Heatmap (Foreground)')
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04, label='Probability')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5,
                target_size=None):
    """
    Predict segmentation mask for an image
    
    Args:
        net: The neural network model
        full_img: PIL Image input
        device: torch device
        scale_factor: Scale factor for input (ignored if target_size is set)
        out_threshold: Threshold for binary classification
        target_size: Tuple (width, height) to resize image before inference
    """
    net.eval()
    # Store original size
    original_size = full_img.size  # (width, height)
    
    # Resize to target size if specified (for models trained on specific size)
    if target_size is not None:
        # Resize image to target size for inference
        resized_img = full_img.resize(target_size, Image.LANCZOS)
        img = torch.from_numpy(BasicDataset.preprocess(None, resized_img, 1.0, is_mask=False))
        print(f"Resized input from {original_size} to {target_size} for inference")
    else:
        # Use original preprocessing with scale_factor
        img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        
        # Always interpolate back to original image size
        output = F.interpolate(output, (original_size[1], original_size[0]), mode='bilinear')
        
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold
        
        # Uncomment to save heatmap visualization
        # sigmoid_heatmap(output, save_path='heatmap.png')
        
    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images (ignored if --target-size is used)')
    parser.add_argument('--target-size', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'),
                        help='Target size (width height) to resize image before inference (e.g., 640 640)')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)
        
        # Determine target size for inference
        target_size = None
        if args.target_size:
            target_size = tuple(args.target_size)  # (width, height)
            logging.info(f'Using target size: {target_size[0]}x{target_size[1]} for inference')

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           target_size=target_size,
                           device=device)
        # print(f'Predicted mask shape: {mask}')

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
