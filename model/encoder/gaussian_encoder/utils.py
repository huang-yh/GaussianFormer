from ...utils.safe_ops import safe_sigmoid, safe_inverse_sigmoid
import torch, torch.nn as nn
from torch import Tensor
from typing import NamedTuple


def spherical2cartesian(anchor, pc_range, phi_activation='loop'):
    if phi_activation == 'sigmoid':
        xyz = safe_sigmoid(anchor[..., :3])
    elif phi_activation == 'loop':
        xy = safe_sigmoid(anchor[..., :2])
        z = torch.remainder(anchor[..., 2:3], 1.0)
        xyz = torch.cat([xy, z], dim=-1)
    else:
        raise NotImplementedError
    rrr = xyz[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    theta = xyz[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    phi = xyz[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
    xxx = rrr * torch.sin(theta) * torch.cos(phi)
    yyy = rrr * torch.sin(theta) * torch.sin(phi)
    zzz = rrr * torch.cos(theta)
    xyz = torch.stack([xxx, yyy, zzz], dim=-1)
    
    return xyz

def cartesian(anchor, pc_range, use_sigmoid=True):
    if use_sigmoid:
        xyz = safe_sigmoid(anchor[..., :3])
    else:
        xyz = anchor[..., :3].clamp(min=1e-6, max=1-1e-6)
    xxx = xyz[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    yyy = xyz[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    zzz = xyz[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
    xyz = torch.stack([xxx, yyy, zzz], dim=-1)
    
    return xyz

def reverse_cartesian(xyz, pc_range, use_sigmoid=True):
    xxx = (xyz[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
    yyy = (xyz[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
    zzz = (xyz[..., 2] - pc_range[2]) / (pc_range[5] - pc_range[2])
    unitxyz = torch.stack([xxx, yyy, zzz], dim=-1)
    if use_sigmoid:
        anchor = safe_inverse_sigmoid(unitxyz)
    else:
        anchor = unitxyz.clamp(min=1e-6, max=1-1e-6)
    return anchor

def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


class GaussianPrediction(NamedTuple):
    means: Tensor
    scales: Tensor
    rotations: Tensor
    opacities: Tensor
    semantics: Tensor
    original_means: Tensor = None
    delta_means: Tensor = None
