import torch
import torch.nn as nn
import torch.nn.functional as F
from .vfm import VisibilityAwareModule
from .spa import SPAModule
from .gpa import SpatialGPA, TemporalGPA

class PICFormer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Model parameters
        c = cfg.model["channel"]
        self.num_joints = cfg.model["num_joints"]
        self.num_frames = cfg.model["num_frames"]
        
        # Input projection layer: project 2D keypoints to feature space
        self.input_projection = nn.Sequential(
            nn.Linear(2, c // 2),
            nn.ReLU(inplace=True),
            nn.Linear(c // 2, c),
            nn.LayerNorm(c)
        )
        
        # Visibility-aware module
        self.vfm = VisibilityAwareModule(
            c, 
            cfg.model["eps"], 
            cfg.model["beta"], 
            cfg.model["tau"]
        )
        
        # Spatial-temporal aggregation module
        self.spa = SPAModule(cfg.model["groups_S"], cfg.model["groups_T"])
        
        # Global position-aware modules
        self.gpa_s = SpatialGPA(c, heads=8)
        self.gpa_t = TemporalGPA(c, heads=8)
        
        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(c * 3, c * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(c * 2, c),
            nn.LayerNorm(c)
        )
        
        # 2D heatmap decoder
        self.heatmap_decoder = nn.Sequential(
            nn.Linear(c, c // 2),
            nn.ReLU(inplace=True),
            nn.Linear(c // 2, 64 * 64)  # 64x64 heatmap
        )
        
        # 3D pose regression head
        self.pose3d_decoder = nn.Sequential(
            nn.Linear(c, c // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(c // 2, c // 4),
            nn.ReLU(inplace=True),
            nn.Linear(c // 4, 3)  # x, y, z coordinates
        )
        
        # Visibility prediction head
        self.visibility_decoder = nn.Sequential(
            nn.Linear(c, c // 2),
            nn.ReLU(inplace=True),
            nn.Linear(c // 2, 1),
            nn.Sigmoid()
        )
        
        # Multiscale feature processing
        self.multiscale_processing = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c, c),
                nn.ReLU(inplace=True),
                nn.Linear(c, c)
            ) for _ in range(3)  # Process 3 scales of features
        ])
        
    def forward(self, X):
        """
        Args:
            X: Input 2D keypoints [B, N, T, 2]
        Returns:
            heatmaps_2d: 2D heatmaps [B, N, T, 64, 64]
            pose_3d: 3D poses [B, N, T, 3]
            visibility: Visibility predictions [B, N, T]
        """
        B, N, T, _ = X.shape
        
        # 1. Input projection
        X_proj = self.input_projection(X)  # [B, N, T, C]
        
        # 2. Visibility-aware module
        X_mod, P_vfm = self.vfm(X_proj)  # [B, N, T, C], [B, N, T]
        
        # 3. Spatial-temporal aggregation module
        multiscale_features, biases = self.spa(X_mod, P_vfm)  # [B, num_scales, T, C], [B, num_scales, T, 1]
        
        # 4. Multiscale feature processing
        processed_features = []
        for i, processor in enumerate(self.multiscale_processing):
            if i < multiscale_features.size(1):
                feat = processor(multiscale_features[:, i, :, :])  # [B, T, C]
                processed_features.append(feat)
        
        # 5. Global position-aware
        # Spatial GPA
        # spatial_bias = biases[:, 0, :, :] if biases.size(1) > 0 else None  # [B, T, 1]
        spatial_features = self.gpa_s(X_mod, None)  # [B, N, T, C]
        
        # Temporal GPA
        # temporal_bias = biases[:, 1, :, :] if biases.size(1) > 1 else None  # [B, N, 1]
        temporal_features = self.gpa_t(X_mod, None)  # [B, N, T, C]
        
        # 6. Feature fusion
        # Expand multiscale features to original dimensions
        if processed_features:
            multiscale_expanded = torch.stack(processed_features, dim=1)  # [B, num_scales, T, C]
            multiscale_expanded = multiscale_expanded.unsqueeze(2).expand(-1, -1, N, -1, -1)  # [B, num_scales, N, T, C]
            multiscale_avg = multiscale_expanded.mean(dim=1)  # [B, N, T, C]
        else:
            multiscale_avg = torch.zeros_like(spatial_features)
        
        # Fuse all features
        fused_features = torch.cat([
            spatial_features, 
            temporal_features, 
            multiscale_avg
        ], dim=-1)  # [B, N, T, C*3]
        
        final_features = self.feature_fusion(fused_features)  # [B, N, T, C]
        
        # 7. Decode outputs
        # 2D heatmaps
        heatmap_logits = self.heatmap_decoder(final_features)  # [B, N, T, 64*64]
        heatmaps_2d = heatmap_logits.view(B, N, T, 64, 64)  # [B, N, T, 64, 64]
        
        # 3D poses
        pose_3d = self.pose3d_decoder(final_features)  # [B, N, T, 3]
        
        # Visibility
        visibility = self.visibility_decoder(final_features).squeeze(-1)  # [B, N, T]
        
        return heatmaps_2d, pose_3d, visibility
    
    def get_multiscale_features(self, X):
        """Get multiscale features for hierarchical consistency loss"""
        B, N, T, _ = X.shape
        
        # Input projection
        X_proj = self.input_projection(X)
        
        # Visibility-aware
        X_mod, _ = self.vfm(X_proj)
        
        # Spatial-temporal aggregation
        multiscale_features, _ = self.spa(X_mod, torch.ones_like(X_mod[:, :, :, 0]))
        
        # Process multiscale features
        processed_features = []
        for i, processor in enumerate(self.multiscale_processing):
            if i < multiscale_features.size(1):
                feat = processor(multiscale_features[:, i, :, :])
                processed_features.append(feat)
        
        return processed_features 