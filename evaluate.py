import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            original_sizes = batch.get('original_size', None)

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)
            
            # If we have original sizes, resize predictions back to original size for evaluation
            if original_sizes is not None:
                # Process each item in the batch
                resized_preds = []
                resized_trues = []
                
                for i in range(len(mask_pred)):
                    orig_size = original_sizes[i]
                    # Resize prediction back to original size
                    pred_resized = F.interpolate(
                        mask_pred[i:i+1], 
                        size=(orig_size[1], orig_size[0]),  # (height, width)
                        mode='bilinear'
                    )
                    # Resize true mask back to original size  
                    true_resized = F.interpolate(
                        mask_true[i:i+1].float().unsqueeze(1), 
                        size=(orig_size[1], orig_size[0]),  # (height, width)
                        mode='nearest'
                    ).squeeze(1).long()
                    
                    resized_preds.append(pred_resized)
                    resized_trues.append(true_resized)
                
                mask_pred = torch.cat(resized_preds, dim=0)
                mask_true = torch.cat(resized_trues, dim=0)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)
