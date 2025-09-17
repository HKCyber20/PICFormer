import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GPAModule(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        assert self.head_dim * heads == dim, "dim must be divisible by heads"
        
        # Multi-head attention layers
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(dim)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V, bias=None):
        """
        Args:
            Q: Query [B, N, T, C]
            K: Key [B, N, T, C]
            V: Value [B, N, T, C]
            bias: Bias [B, N, T, 1] or None
        Returns:
            output: Attention output [B, N, T, C]
        """
        B, N, T, C = Q.shape
        
        # Add positional encoding
        Q = self.pos_encoding(Q)
        K = self.pos_encoding(K)
        V = self.pos_encoding(V)
        
        # Multi-head attention computation
        Q = self.q_proj(Q).view(B, N, T, self.heads, self.head_dim).transpose(2, 3)  # [B, N, H, T, D]
        K = self.k_proj(K).view(B, N, T, self.heads, self.head_dim).transpose(2, 3)  # [B, N, H, T, D]
        V = self.v_proj(V).view(B, N, T, self.heads, self.head_dim).transpose(2, 3)  # [B, N, H, T, D]
        
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, N, H, T, T]
        
        # Add bias (if provided)
        if bias is not None:
            # Ensure bias dimensions are correct
            if len(bias.shape) == 3:  # [B, N, T]
                bias = bias.unsqueeze(-1)  # [B, N, T, 1]
            bias = bias.unsqueeze(2).expand(-1, -1, self.heads, -1, -1)  # [B, N, H, T, 1]
            scores = scores + bias
        
        # Apply attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Calculate output
        output = torch.matmul(attn_weights, V)  # [B, N, H, T, D]
        output = output.transpose(2, 3).contiguous().view(B, N, T, C)  # [B, N, T, C]
        output = self.out_proj(output)
        
        # Residual connection and layer normalization
        output = self.norm1(output + Q.reshape(B, N, T, C))
        
        # Feed Forward Network
        ffn_output = self.ffn(output)
        output = self.norm2(output + ffn_output)
        
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(0)  # [1, 1, T, C]
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, N, T, C]
        Returns:
            x + pe: Tensor with added positional encoding
        """
        return x + self.pe[:, :, :x.size(2), :]

class SpatialGPA(GPAModule):
    """Spatial Global Position-Aware module"""
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__(dim, heads, dropout)
        
    def forward(self, X, bias=None):
        """
        Args:
            X: Input features [B, N, T, C]
            bias: Spatial bias [B, N, T, 1]
        Returns:
            output: Spatial attention output [B, N, T, C]
        """
        return super().forward(X, X, X, bias)

class TemporalGPA(GPAModule):
    """Temporal Global Position-Aware module"""
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__(dim, heads, dropout)
        
    def forward(self, X, bias=None):
        """
        Args:
            X: Input features [B, N, T, C]
            bias: Temporal bias [B, N, T, 1]
        Returns:
            output: Temporal attention output [B, N, T, C]
        """
        # Transpose to apply attention on temporal dimension
        X_t = X.transpose(1, 2)  # [B, T, N, C]
        bias_t = bias.transpose(1, 2) if bias is not None else None  # [B, T, N, 1]
        
        output_t = super().forward(X_t, X_t, X_t, bias_t)
        output = output_t.transpose(1, 2)  # [B, N, T, C]
        
        return output 