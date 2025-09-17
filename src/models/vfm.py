import torch
import torch.nn as nn

class VisibilityAwareModule(nn.Module):
    def __init__(self, channel, eps=0.01, beta=10.0, tau=0.5):
        super().__init__()
        self.channel = channel
        self.eps = eps
        self.beta = beta
        self.tau = tau
        
        # Visibility prediction MLP
        self.visibility_mlp = nn.Sequential(
            nn.Linear(channel, channel // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(channel // 2, channel // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channel // 4, 1)
        )
        
        # Feature modulation network
        self.feature_modulation = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel)
        )
        
    def forward(self, X):
        """
        Args:
            X: Input features [B, N, T, C]
        Returns:
            X_mod: Modulated features [B, N, T, C]
            P: Visibility probabilities [B, N, T]
        """
        B, N, T, C = X.shape
        
        # Reshape to [B*N*T, C] for batch processing
        X_flat = X.view(B * N * T, C)
        
        # Predict visibility probabilities
        P_logits = self.visibility_mlp(X_flat)  # [B*N*T, 1]
        P = torch.sigmoid(P_logits).view(B, N, T)  # [B, N, T]
        
        # Calculate visibility weights
        V = self.eps + (1 - self.eps) * torch.sigmoid(
            self.beta * (P - self.tau)
        )  # [B, N, T]
        
        # Feature modulation
        X_mod_flat = self.feature_modulation(X_flat)  # [B*N*T, C]
        X_mod_flat = X_mod_flat.view(B, N, T, C)  # [B, N, T, C]
        
        # Apply visibility weights
        V_expanded = V.unsqueeze(-1)  # [B, N, T, 1]
        X_mod = V_expanded * X_mod_flat + (1 - V_expanded) * X
        
        return X_mod, P 