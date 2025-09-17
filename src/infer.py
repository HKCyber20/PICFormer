import os
import torch
import numpy as np
import argparse
import json
from tqdm import tqdm

from config import cfg, parse_args
from models.picformer import PICFormer
from utils import normalize_2d_poses, denormalize_2d_poses

def load_model(checkpoint_path, device):
    """Load trained model"""
    model = PICFormer(cfg).to(device)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint file not found {checkpoint_path}")
    
    model.eval()
    return model

def preprocess_2d_poses(poses_2d, image_size=(256, 256)):
    """Preprocess 2D pose data"""
    # Ensure correct input format
    if len(poses_2d.shape) == 3:  # [N, T, 2]
        poses_2d = poses_2d.unsqueeze(0)  # Add batch dimension [1, N, T, 2]
    
    # Normalize to [-1, 1] range
    poses_normalized = normalize_2d_poses(poses_2d, image_size)
    
    return poses_normalized

def postprocess_results(pose_3d, visibility, heatmaps_2d, image_size=(256, 256)):
    """Postprocess inference results"""
    # Convert results to numpy arrays
    pose_3d = pose_3d.cpu().numpy()
    visibility = visibility.cpu().numpy()
    heatmaps_2d = heatmaps_2d.cpu().numpy()
    
    # Remove batch dimension (if exists)
    if len(pose_3d.shape) == 4:
        pose_3d = pose_3d.squeeze(0)  # [N, T, 3]
    if len(visibility.shape) == 3:
        visibility = visibility.squeeze(0)  # [N, T]
    if len(heatmaps_2d.shape) == 5:
        heatmaps_2d = heatmaps_2d.squeeze(0)  # [N, T, 64, 64]
    
    return {
        'pose_3d': pose_3d,
        'visibility': visibility,
        'heatmaps_2d': heatmaps_2d
    }

def infer_single_sequence(model, poses_2d, device, image_size=(256, 256)):
    """Infer single sequence"""
    # Preprocessing
    poses_2d_processed = preprocess_2d_poses(poses_2d, image_size)
    poses_2d_processed = poses_2d_processed.to(device)
    
    # Inference
    with torch.no_grad():
        heatmaps_2d, pose_3d, visibility = model(poses_2d_processed)
    
    # Postprocessing
    results = postprocess_results(pose_3d, visibility, heatmaps_2d, image_size)
    
    return results

def infer_batch(model, poses_2d_batch, device, image_size=(256, 256)):
    """Batch inference"""
    results_batch = []
    
    for i, poses_2d in enumerate(tqdm(poses_2d_batch, desc="Inferring")):
        results = infer_single_sequence(model, poses_2d, device, image_size)
        results_batch.append(results)
    
    return results_batch

def save_results(results, save_path):
    """Save inference results"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if isinstance(results, list):
        # Batch results
        np.savez_compressed(
            save_path,
            pose_3d=[r['pose_3d'] for r in results],
            visibility=[r['visibility'] for r in results],
            heatmaps_2d=[r['heatmaps_2d'] for r in results]
        )
    else:
        # Single result
        np.savez_compressed(
            save_path,
            pose_3d=results['pose_3d'],
            visibility=results['visibility'],
            heatmaps_2d=results['heatmaps_2d']
        )
    
    print(f"Results saved to: {save_path}")

def load_2d_poses_from_file(file_path):
    """Load 2D pose data from file"""
    if file_path.endswith('.npy'):
        poses_2d = np.load(file_path)
    elif file_path.endswith('.npz'):
        data = np.load(file_path)
        poses_2d = data['poses_2d'] if 'poses_2d' in data else data['arr_0']
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
            poses_2d = np.array(data['poses_2d'])
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    return torch.FloatTensor(poses_2d)

def main():
    parser = argparse.ArgumentParser(description='PICFormer Inference')
    parser.add_argument('--input', type=str, required=True, help='Input 2D pose file path')
    parser.add_argument('--output', type=str, default='output_results.npz', help='Output results file path')
    parser.add_argument('--checkpoint', type=str, default='checkpoint/best.pth', help='Model checkpoint path')
    parser.add_argument('--batch', action='store_true', help='Whether to process in batch')
    parser.add_argument('--image_size', type=int, nargs=2, default=[256, 256], help='Image size [width, height]')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(cfg.train["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Load input data
    print(f"Loading input data: {args.input}")
    poses_2d_data = load_2d_poses_from_file(args.input)
    
    # Inference
    if args.batch and len(poses_2d_data.shape) == 4:  # [B, N, T, 2]
        print("Batch inference mode")
        results = infer_batch(model, poses_2d_data, device, tuple(args.image_size))
    else:
        print("Single sequence inference mode")
        results = infer_single_sequence(model, poses_2d_data, device, tuple(args.image_size))
    
    # Save results
    save_results(results, args.output)
    
    # Print result summary
    if isinstance(results, list):
        print(f"Batch inference completed, processed {len(results)} sequences")
        for i, result in enumerate(results):
            print(f"Sequence {i}: 3D pose shape {result['pose_3d'].shape}, visibility shape {result['visibility'].shape}")
    else:
        print(f"Inference completed: 3D pose shape {results['pose_3d'].shape}, visibility shape {results['visibility'].shape}")

if __name__ == "__main__":
    main() 