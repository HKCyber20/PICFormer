import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse

from config import cfg, parse_args
from datasets import get_dataloader
from models.picformer import PICFormer
from utils import (
    mpjpe_loss, pampjpe_loss, bce_loss, hc_loss, heatmap_loss,
    setup_logger, save_checkpoint, load_checkpoint, compute_accuracy
)

def train_epoch(model, train_loader, optimizer, criterion, device, epoch, logger, writer):
    """Train one epoch"""
    model.train()
    total_loss = 0.0
    total_mpjpe = 0.0
    total_visibility_acc = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (X2d, Y3d, V_gt) in enumerate(pbar):
        X2d, Y3d, V_gt = X2d.to(device), Y3d.to(device), V_gt.to(device)
        
        # Forward pass
        heatmaps_2d, pose_3d, visibility = model(X2d)
        
        # Calculate losses
        # MPJPE loss
        mpjpe = mpjpe_loss(pose_3d, Y3d)
        
        # Visibility loss
        vis_loss = bce_loss(visibility, V_gt)
        
        # Heatmap loss (if 2D heatmap labels are available)
        heatmap_loss_val = heatmap_loss(heatmaps_2d, torch.zeros_like(heatmaps_2d))  # Simplified processing
        
        # Hierarchical consistency loss
        multiscale_preds = model.get_multiscale_features(X2d)
        multiscale_gt = [Y3d.mean(dim=1)] * len(multiscale_preds)  # Simplified processing
        hc_loss_val = hc_loss(multiscale_preds, multiscale_gt)
        
        # Total loss
        total_loss_batch = (
            cfg.loss["lambda_mpjpe"] * mpjpe +
            cfg.loss["lambda_vis"] * vis_loss +
            cfg.loss["lambda_hc"] * hc_loss_val +
            heatmap_loss_val
        )
        
        # Backward pass
        optimizer.zero_grad()
        total_loss_batch.backward()
        optimizer.step()
        
        # Statistics
        total_loss += total_loss_batch.item()
        total_mpjpe += mpjpe.item()
        total_visibility_acc += compute_accuracy(visibility, V_gt).item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{total_loss_batch.item():.4f}',
            'MPJPE': f'{mpjpe.item():.4f}',
            'Vis_Acc': f'{compute_accuracy(visibility, V_gt).item():.4f}'
        })
        
        # Log to TensorBoard
        if batch_idx % cfg.train["log_interval"] == 0:
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Loss', total_loss_batch.item(), step)
            writer.add_scalar('Train/MPJPE', mpjpe.item(), step)
            writer.add_scalar('Train/Visibility_Accuracy', compute_accuracy(visibility, V_gt).item(), step)
    
    # Calculate averages
    avg_loss = total_loss / num_batches
    avg_mpjpe = total_mpjpe / num_batches
    avg_visibility_acc = total_visibility_acc / num_batches
    
    logger.info(f'Epoch {epoch} - Train Loss: {avg_loss:.4f}, MPJPE: {avg_mpjpe:.4f}, Vis_Acc: {avg_visibility_acc:.4f}')
    
    return avg_loss, avg_mpjpe, avg_visibility_acc

def validate(model, val_loader, criterion, device, epoch, logger, writer):
    """Validation"""
    model.eval()
    total_loss = 0.0
    total_mpjpe = 0.0
    total_pampjpe = 0.0
    total_visibility_acc = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (X2d, Y3d, V_gt) in enumerate(tqdm(val_loader, desc='Validation')):
            X2d, Y3d, V_gt = X2d.to(device), Y3d.to(device), V_gt.to(device)
            
            # Forward pass
            heatmaps_2d, pose_3d, visibility = model(X2d)
            
            # Calculate losses
            mpjpe = mpjpe_loss(pose_3d, Y3d)
            pampjpe = pampjpe_loss(pose_3d, Y3d)
            vis_loss = bce_loss(visibility, V_gt)
            
            # Total loss
            total_loss_batch = (
                cfg.loss["lambda_mpjpe"] * mpjpe +
                cfg.loss["lambda_vis"] * vis_loss
            )
            
            # Statistics
            total_loss += total_loss_batch.item()
            total_mpjpe += mpjpe.item()
            total_pampjpe += pampjpe.item()
            total_visibility_acc += compute_accuracy(visibility, V_gt).item()
            num_batches += 1
    
    # Calculate averages
    avg_loss = total_loss / num_batches
    avg_mpjpe = total_mpjpe / num_batches
    avg_pampjpe = total_pampjpe / num_batches
    avg_visibility_acc = total_visibility_acc / num_batches
    
    logger.info(f'Epoch {epoch} - Val Loss: {avg_loss:.4f}, MPJPE: {avg_mpjpe:.4f}, PA-MPJPE: {avg_pampjpe:.4f}, Vis_Acc: {avg_visibility_acc:.4f}')
    
    # Log to TensorBoard
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    writer.add_scalar('Val/MPJPE', avg_mpjpe, epoch)
    writer.add_scalar('Val/PA-MPJPE', avg_pampjpe, epoch)
    writer.add_scalar('Val/Visibility_Accuracy', avg_visibility_acc, epoch)
    
    return avg_loss, avg_mpjpe, avg_pampjpe, avg_visibility_acc

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(cfg.train["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup logger
    logger = setup_logger()
    
    # Setup TensorBoard
    writer = SummaryWriter('runs/picformer_training')
    
    # Create model
    model = PICFormer(cfg).to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.train["lr"],
        weight_decay=cfg.train["weight_decay"]
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Create data loaders
    train_loader = get_dataloader(cfg, "train")
    val_loader = get_dataloader(cfg, "val")
    
    # Resume training (if checkpoint exists)
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Loading checkpoint from {args.resume}")
        start_epoch, best_val_loss = load_checkpoint(model, optimizer, args.resume)
        start_epoch += 1
    
    # Training loop
    for epoch in range(start_epoch, cfg.train["epochs"]):
        # Training
        train_loss, train_mpjpe, train_vis_acc = train_epoch(
            model, train_loader, optimizer, None, device, epoch, logger, writer
        )
        
        # Validation
        val_loss, val_mpjpe, val_pampjpe, val_vis_acc = validate(
            model, val_loader, None, device, epoch, logger, writer
        )
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(args.checkpoint, 'best.pth')
            )
            logger.info(f"Saved best model with validation loss: {val_loss:.4f}")
        
        # Periodic checkpoint saving
        if epoch % cfg.train["save_interval"] == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(args.checkpoint, f'checkpoint_epoch_{epoch}.pth')
            )
    
    # Save final model
    save_checkpoint(
        model, optimizer, cfg.train["epochs"] - 1, val_loss,
        os.path.join(args.checkpoint, 'final.pth')
    )
    
    logger.info("Training completed!")
    writer.close()

if __name__ == "__main__":
    main() 