import torch
import torch.nn as nn
import torch.nn.functional as F

class SPAModule(nn.Module):
    def __init__(self, groups_S, groups_T, alpha=0.5):
        super().__init__()
        self.groups_S = groups_S  # Spatial groups
        self.groups_T = groups_T  # Temporal groups
        self.alpha = alpha
        
        # Spatial aggregation networks
        self.spatial_aggregation = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(group), len(group)),
                nn.ReLU(inplace=True),
                nn.Linear(len(group), 1)
            ) for group in groups_S
        ])
        
        # Temporal aggregation networks
        self.temporal_aggregation = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(group), len(group)),
                nn.ReLU(inplace=True),
                nn.Linear(len(group), 1)
            ) for group in groups_T
        ])
        
        # Multiscale feature fusion
        total_scales = len(groups_S) + len(groups_T)
        self.multiscale_fusion = nn.Sequential(
            nn.Linear(total_scales, total_scales),
            nn.ReLU(inplace=True),
            nn.Linear(total_scales, 1)
        )
        
    def aggregate(self, H, P, groups, aggregation_nets):
        """
        Perform group aggregation
        Args:
            H: Features [B, N, T, C]
            P: Visibility probabilities [B, N, T]
            groups: Group list
            aggregation_nets: Aggregation network list
        Returns:
            aggregated_features: Aggregated features [B, num_groups, T, C]
            biases: Biases [B, num_groups, T, 1]
        """
        B, N, T, C = H.shape
        num_groups = len(groups)
        
        aggregated_features = []
        biases = []
        
        for i, group in enumerate(groups):
            # Extract features for current group
            group_features = H[:, group, :, :]  # [B, group_size, T, C]
            group_visibility = P[:, group, :]   # [B, group_size, T]
            
            # Calculate weighted average
            weights = F.softmax(group_visibility, dim=1)  # [B, group_size, T]
            weights = weights.unsqueeze(-1)  # [B, group_size, T, 1]
            
            # Aggregate features
            weighted_features = group_features * weights  # [B, group_size, T, C]
            aggregated_feat = weighted_features.sum(dim=1)  # [B, T, C]
            
            # Calculate bias
            bias_input = group_visibility  # [B, group_size, T]
            bias_input = bias_input.transpose(1, 2)  # [B, T, group_size]
            bias = aggregation_nets[i](bias_input)  # [B, T, 1]
            
            aggregated_features.append(aggregated_feat)
            biases.append(bias)
        
        # Stack features and biases from all groups
        aggregated_features = torch.stack(aggregated_features, dim=1)  # [B, num_groups, T, C]
        biases = torch.stack(biases, dim=1)  # [B, num_groups, T, 1]
        
        return aggregated_features, biases
    
    def forward(self, X, P):
        """
        Args:
            X: Input features [B, N, T, C]
            P: Visibility probabilities [B, N, T]
        Returns:
            multiscale_features: Multiscale features [B, num_scales, T, C]
            biases: Biases [B, num_scales, T, 1]
        """
        B, N, T, C = X.shape
        
        # Spatial aggregation
        spatial_features, spatial_biases = self.aggregate(
            X, P, self.groups_S, self.spatial_aggregation
        )
        
        # Temporal aggregation
        temporal_features, temporal_biases = self.aggregate(
            X, P, self.groups_T, self.temporal_aggregation
        )
        
        # Combine spatial and temporal features
        multiscale_features = torch.cat([spatial_features, temporal_features], dim=1)
        multiscale_biases = torch.cat([spatial_biases, temporal_biases], dim=1)
        
        # Multiscale feature fusion
        fusion_input = multiscale_biases.squeeze(-1)  # [B, num_scales, T]
        # Calculate fusion weights for each time step separately
        fusion_weights = []
        for t in range(fusion_input.size(2)):
            t_input = fusion_input[:, :, t]  # [B, num_scales]
            t_weights = F.softmax(self.multiscale_fusion(t_input), dim=1)  # [B, num_scales]
            fusion_weights.append(t_weights)
        fusion_weights = torch.stack(fusion_weights, dim=2)  # [B, num_scales, T]
        fusion_weights = fusion_weights.unsqueeze(-1)  # [B, num_scales, T, 1]
        
        # Weighted fusion
        fused_features = (multiscale_features * fusion_weights).sum(dim=1, keepdim=True)  # [B, 1, T, C]
        
        # Final output includes original multiscale features and fused features
        final_features = torch.cat([multiscale_features, fused_features], dim=1)
        final_biases = torch.cat([multiscale_biases, fusion_weights], dim=1)
        
        return final_features, final_biases 