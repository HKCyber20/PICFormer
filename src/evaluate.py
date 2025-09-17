import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
import json

from config import cfg, parse_args
from datasets import get_dataloader
from models.picformer import PICFormer
from utils import mpjpe_loss, pampjpe_loss, compute_accuracy

def evaluate_model(model, data_loader, device):
    """Evaluate model performance"""
    model.eval()
    
    total_mpjpe = 0.0
    total_pampjpe = 0.0
    total_visibility_acc = 0.0
    num_samples = 0
    
    # Store detailed results
    results = {
        'mpjpe_list': [],
        'pampjpe_list': [],
        'visibility_acc_list': [],
        'predictions': [],
        'ground_truth': []
    }
    
    with torch.no_grad():
        for batch_idx, (X2d, Y3d, V_gt) in enumerate(tqdm(data_loader, desc='Evaluating')):
            X2d, Y3d, V_gt = X2d.to(device), Y3d.to(device), V_gt.to(device)
            
            # Forward pass
            heatmaps_2d, pose_3d, visibility = model(X2d)
            
            # Calculate metrics
            batch_size = X2d.size(0)
            
            # MPJPE
            mpjpe = mpjpe_loss(pose_3d, Y3d)
            total_mpjpe += mpjpe.item() * batch_size
            
            # PA-MPJPE
            pampjpe = pampjpe_loss(pose_3d, Y3d)
            total_pampjpe += pampjpe.item() * batch_size
            
            # Visibility accuracy
            vis_acc = compute_accuracy(visibility, V_gt)
            total_visibility_acc += vis_acc.item() * batch_size
            
            num_samples += batch_size
            
            # Store detailed results
            results['mpjpe_list'].extend([mpjpe.item()] * batch_size)
            results['pampjpe_list'].extend([pampjpe.item()] * batch_size)
            results['visibility_acc_list'].extend([vis_acc.item()] * batch_size)
            
            # Store predictions and ground truth (for further analysis)
            results['predictions'].append({
                'pose_3d': pose_3d.cpu().numpy(),
                'visibility': visibility.cpu().numpy(),
                'heatmaps_2d': heatmaps_2d.cpu().numpy()
            })
            results['ground_truth'].append({
                'pose_3d': Y3d.cpu().numpy(),
                'visibility': V_gt.cpu().numpy()
            })
    
    # Calculate averages
    avg_mpjpe = total_mpjpe / num_samples
    avg_pampjpe = total_pampjpe / num_samples
    avg_visibility_acc = total_visibility_acc / num_samples
    
    # Calculate standard deviations
    std_mpjpe = np.std(results['mpjpe_list'])
    std_pampjpe = np.std(results['pampjpe_list'])
    std_visibility_acc = np.std(results['visibility_acc_list'])
    
    return {
        'avg_mpjpe': avg_mpjpe,
        'avg_pampjpe': avg_pampjpe,
        'avg_visibility_acc': avg_visibility_acc,
        'std_mpjpe': std_mpjpe,
        'std_pampjpe': std_pampjpe,
        'std_visibility_acc': std_visibility_acc,
        'num_samples': num_samples,
        'detailed_results': results
    }

def print_results(results):
    """Print evaluation results"""
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Number of samples: {results['num_samples']}")
    print(f"MPJPE: {results['avg_mpjpe']:.4f} ± {results['std_mpjpe']:.4f}")
    print(f"PA-MPJPE: {results['avg_pampjpe']:.4f} ± {results['std_pampjpe']:.4f}")
    print(f"Visibility Accuracy: {results['avg_visibility_acc']:.4f} ± {results['std_visibility_acc']:.4f}")
    print("="*50)

def save_results(results, save_path):
    """Save evaluation results"""
    # Create save directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save main metrics
    metrics = {
        'avg_mpjpe': results['avg_mpjpe'],
        'avg_pampjpe': results['avg_pampjpe'],
        'avg_visibility_acc': results['avg_visibility_acc'],
        'std_mpjpe': results['std_mpjpe'],
        'std_pampjpe': results['std_pampjpe'],
        'std_visibility_acc': results['std_visibility_acc'],
        'num_samples': results['num_samples']
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"Evaluation results saved to: {save_path}")

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(cfg.train["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = PICFormer(cfg).to(device)
    
    # Load trained model
    checkpoint_path = os.path.join(args.checkpoint, 'best.pth')
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found {checkpoint_path}")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create data loader
    val_loader = get_dataloader(cfg, "val")
    
    # Evaluate model
    print("Starting evaluation...")
    results = evaluate_model(model, val_loader, device)
    
    # Print results
    print_results(results)
    
    # Save results
    save_path = os.path.join(args.checkpoint, 'evaluation_results.json')
    save_results(results, save_path)
    
    # Optional: save detailed results (may be large)
    detailed_save_path = os.path.join(args.checkpoint, 'detailed_results.npz')
    np.savez_compressed(
        detailed_save_path,
        mpjpe_list=np.array(results['detailed_results']['mpjpe_list']),
        pampjpe_list=np.array(results['detailed_results']['pampjpe_list']),
        visibility_acc_list=np.array(results['detailed_results']['visibility_acc_list'])
    )
    print(f"Detailed results saved to: {detailed_save_path}")

if __name__ == "__main__":
    main() 