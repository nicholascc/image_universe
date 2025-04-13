import torch
import math
import torch.nn as nn

class SinusoidalEmb(nn.Module):
    """
    Returns a sinusoidal embedding for an integer t.
    Typical dimension: embed_dim = 128 or 256.
    """
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t):
        """
        t: (batch,)  integers in [0..T].
        Return shape: (batch, embed_dim).
        """
        # For example, compute frequencies from 1 to embed_dim/2
        half_dim = self.embed_dim // 2
        freq = torch.exp(
            torch.arange(half_dim, device=t.device) * (-math.log(10000) / (half_dim - 1))
        )
        # t is shape (batch,), unsqueeze to (batch,1)
        t_ = t.unsqueeze(1).float()
        # shape => (batch, half_dim)
        sinusoidal_inp = t_ * freq
        # Now cat sin and cos
        sin = torch.sin(sinusoidal_inp)
        cos = torch.cos(sinusoidal_inp)
        emb = torch.cat([sin, cos], dim=1)  # (batch, embed_dim)
        return emb

class TimeEmbed(nn.Module):
    """
    A small MLP to project the sinusoidal embedding to a chosen dimension.
    For instance, we can produce something of size 'time_emb_dim' which
    we later broadcast + add into feature maps.
    """
    def __init__(self, embed_dim=128, time_emb_dim=256):
        super().__init__()
        self.sin_emb = SinusoidalEmb(embed_dim=embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

    def forward(self, t):
        emb = self.sin_emb(t)   # shape (B, embed_dim)
        emb = self.mlp(emb)     # shape (B, time_emb_dim)
        return emb

