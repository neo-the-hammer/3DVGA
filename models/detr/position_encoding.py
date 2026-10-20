import math
import torch
from torch import nn


class PositionEmbeddingSine3D(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        if scale is None:
            scale = 2 * math.pi * 32
        self.scale = scale

    def forward(self, xyz):
        B, N, C = xyz.shape  # C=3
        xyz = xyz * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=xyz.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_dim = []
        for i in range(C):
            pos_embd_dim = xyz[:, :, i, None].repeat(1, 1, self.num_pos_feats) / dim_t
            pos_embd_dim = torch.cat((pos_embd_dim[:, :, 0::2].sin(), pos_embd_dim[:, :, 1::2].cos()), dim=-1)
            pos_dim.append(pos_embd_dim.contiguous())
        val_xyz = torch.cat(pos_dim, dim=-1)
        return val_xyz


def build_position_encoding(position_embedding, hidden_dim, input_dim, scale=None):
    N_steps = hidden_dim // input_dim
    assert hidden_dim % input_dim == 0, 'position encoding not divisable by input_dim'
    assert N_steps > 0, 'you should have position encoding'
    if position_embedding in ('sine'):
        position_embedding = PositionEmbeddingSine3D(num_pos_feats=N_steps, scale=scale)
    elif position_embedding in ('learned'):
        pass
    else:
        raise ValueError(f"not supported {position_embedding}")

    return position_embedding
