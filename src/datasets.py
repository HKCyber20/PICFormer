import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py

class Human36M(Dataset):
    def __init__(self, cfg, split="train", data_path=None):
        self.cfg = cfg
        self.split = split
        self.data_path = data_path or cfg.data["data_path"]
        
        # Define number of joints
        self.num_joints = cfg.model["num_joints"]
        self.num_frames = cfg.model["num_frames"]
        
        # Load data
        self.data_2d = []
        self.data_3d = []
        self.visibility = []
        
        self._load_data()
        
    def _load_data(self):
        """Load Human36M dataset"""
        # This should load data according to actual data format
        # Example: assume data is stored in h5 files
        data_file = os.path.join(self.data_path, f"h36m_{self.split}.h5")
        
        if os.path.exists(data_file):
            with h5py.File(data_file, 'r') as f:
                self.data_2d = f['poses_2d'][:]
                self.data_3d = f['poses_3d'][:]
                self.visibility = f['visibility'][:]
        else:
            # If no data file exists, create dummy data
            print(f"Warning: {data_file} not found, creating dummy data")
            num_samples = 1000 if self.split == "train" else 200
            self.data_2d = np.random.randn(num_samples, self.num_frames, self.num_joints, 2)
            self.data_3d = np.random.randn(num_samples, self.num_frames, self.num_joints, 3)
            self.visibility = np.random.randint(0, 2, (num_samples, self.num_frames, self.num_joints))
    
    def __len__(self):
        return len(self.data_2d)
    
    def __getitem__(self, idx):
        # Get 2D keypoints
        X2d = torch.FloatTensor(self.data_2d[idx])  # [T, N, 2]
        
        # Get 3D keypoints
        Y3d = torch.FloatTensor(self.data_3d[idx])  # [T, N, 3]
        
        # Get visibility labels
        visibility_gt = torch.FloatTensor(self.visibility[idx])  # [T, N]
        
        # Adjust dimension order to [N, T, C]
        X2d = X2d.permute(1, 0, 2)  # [N, T, 2]
        Y3d = Y3d.permute(1, 0, 2)  # [N, T, 3]
        visibility_gt = visibility_gt.permute(1, 0)  # [N, T]
        
        return X2d, Y3d, visibility_gt

class ThreeDPW(Dataset):
    def __init__(self, cfg, split="val", data_path=None):
        self.cfg = cfg
        self.split = split
        self.data_path = data_path or cfg.data["data_path"]
        
        self.num_joints = cfg.model["num_joints"]
        self.num_frames = cfg.model["num_frames"]
        
        self.data_2d = []
        self.data_3d = []
        self.visibility = []
        
        self._load_data()
    
    def _load_data(self):
        """Load 3DPW dataset"""
        data_file = os.path.join(self.data_path, f"3dpw_{self.split}.h5")
        
        if os.path.exists(data_file):
            with h5py.File(data_file, 'r') as f:
                self.data_2d = f['poses_2d'][:]
                self.data_3d = f['poses_3d'][:]
                self.visibility = f['visibility'][:]
        else:
            # Create dummy data
            print(f"Warning: {data_file} not found, creating dummy data")
            num_samples = 200
            self.data_2d = np.random.randn(num_samples, self.num_frames, self.num_joints, 2)
            self.data_3d = np.random.randn(num_samples, self.num_frames, self.num_joints, 3)
            self.visibility = np.random.randint(0, 2, (num_samples, self.num_frames, self.num_joints))
    
    def __len__(self):
        return len(self.data_2d)
    
    def __getitem__(self, idx):
        X2d = torch.FloatTensor(self.data_2d[idx])
        Y3d = torch.FloatTensor(self.data_3d[idx])
        visibility_gt = torch.FloatTensor(self.visibility[idx])
        
        # Adjust dimension order
        X2d = X2d.permute(1, 0, 2)
        Y3d = Y3d.permute(1, 0, 2)
        visibility_gt = visibility_gt.permute(1, 0)
        
        return X2d, Y3d, visibility_gt

def get_dataloader(cfg, split="train"):
    """Get data loader"""
    if split == "train":
        dataset = Human36M(cfg, split)
    else:
        dataset = ThreeDPW(cfg, split)
    
    return DataLoader(
        dataset,
        batch_size=cfg.data["batch_size"],
        shuffle=(split == "train"),
        num_workers=cfg.data["num_workers"],
        pin_memory=True
    ) 