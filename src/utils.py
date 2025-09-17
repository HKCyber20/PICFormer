import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
from datetime import datetime

def mpjpe_loss(pred, target):
    """Calculate Mean Per Joint Position Error"""
    return torch.mean(torch.norm(pred - target, dim=-1))

def pampjpe_loss(pred, target):
    """Calculate Procrustes Aligned MPJPE"""
    # Simplified PA-MPJPE implementation
    # In practice, more complex Procrustes alignment is needed
    pred_centered = pred - pred.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)
    
    # Calculate scale factors
    pred_scale = torch.norm(pred_centered, dim=-1).mean(dim=1, keepdim=True)
    target_scale = torch.norm(target_centered, dim=-1).mean(dim=1, keepdim=True)
    
    # Scale alignment
    pred_aligned = pred_centered * (target_scale / pred_scale).unsqueeze(-1)
    
    return torch.mean(torch.norm(pred_aligned - target_centered, dim=-1))

def bce_loss(pred_vis, gt_vis):
    """Binary cross entropy loss for visibility prediction"""
    loss_fn = nn.BCELoss()
    return loss_fn(pred_vis, gt_vis)

def hc_loss(multiscale_preds, multiscale_gt):
    """Hierarchical consistency loss"""
    loss = 0.0
    for i in range(len(multiscale_preds) - 1):
        # Calculate differences between adjacent scales
        pred_diff = multiscale_preds[i] - multiscale_preds[i+1]
        gt_diff = multiscale_gt[i] - multiscale_gt[i+1]
        
        # Calculate consistency loss of differences
        loss += torch.mean(torch.norm(pred_diff - gt_diff, dim=-1))
    
    return loss

def heatmap_loss(pred_heatmaps, gt_heatmaps):
    """Heatmap loss"""
    return F.mse_loss(pred_heatmaps, gt_heatmaps)

def setup_logger(log_dir="logs"):
    """Setup logger"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """Save checkpoint"""
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

def compute_accuracy(pred_vis, gt_vis):
    """Calculate visibility prediction accuracy"""
    pred_binary = (pred_vis > 0.5).float()
    correct = (pred_binary == gt_vis).float()
    accuracy = correct.mean()
    return accuracy

def normalize_2d_poses(poses_2d, image_size=(256, 256)):
    """Normalize 2D poses to [-1, 1] range"""
    h, w = image_size
    poses_normalized = poses_2d.clone()
    poses_normalized[..., 0] = (poses_normalized[..., 0] / w) * 2 - 1  # x coordinate
    poses_normalized[..., 1] = (poses_normalized[..., 1] / h) * 2 - 1  # y coordinate
    return poses_normalized

def denormalize_2d_poses(poses_2d, image_size=(256, 256)):
    """Denormalize 2D poses"""
    h, w = image_size
    poses_denormalized = poses_2d.clone()
    poses_denormalized[..., 0] = (poses_denormalized[..., 0] + 1) / 2 * w  # x coordinate
    poses_denormalized[..., 1] = (poses_denormalized[..., 1] + 1) / 2 * h  # y coordinate
    return poses_denormalized

def create_heatmaps(poses_2d, image_size=(64, 64), sigma=1.0):
    """Create heatmaps from 2D keypoints"""
    batch_size, num_joints, _ = poses_2d.shape
    h, w = image_size
    
    heatmaps = torch.zeros(batch_size, num_joints, h, w)
    
    for b in range(batch_size):
        for j in range(num_joints):
            x, y = poses_2d[b, j]
            
            # Convert coordinates to heatmap coordinate system
            x = (x + 1) / 2 * w
            y = (y + 1) / 2 * h
            
            # Create Gaussian heatmap
            for i in range(h):
                for k in range(w):
                    dist = ((i - y) ** 2 + (k - x) ** 2) / (2 * sigma ** 2)
                    heatmaps[b, j, i, k] = torch.exp(-dist)
    
    return heatmaps 