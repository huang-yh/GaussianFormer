from mmengine.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn import Scale
from functools import partial
import torch.nn as nn, torch
import torch.nn.functional as F
from .utils import linear_relu_ln, GaussianPrediction, cartesian, reverse_cartesian
from ...utils.safe_ops import safe_sigmoid


@MODELS.register_module()
class SparseGaussian3DRefinementModuleV2(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        pc_range=None,
        scale_range=None,
        unit_xyz=None,
        semantics=False,
        semantic_dim=None,
        include_opa=True,
        semantics_activation='softmax',
        xyz_activation="sigmoid",
        scale_activation="sigmoid",
        **kwargs,
    ):
        super().__init__()
        self.embed_dims = embed_dims

        if semantics:
            assert semantic_dim is not None
        else:
            semantic_dim = 0
                
        self.output_dim = 10 + int(include_opa) + semantic_dim
        self.semantic_start = 10 + int(include_opa)
        self.semantic_dim = semantic_dim
        self.include_opa = include_opa
        self.semantics_activation = semantics_activation
        self.xyz_act = xyz_activation
        self.scale_act = scale_activation

        self.pc_range = pc_range
        self.scale_range = scale_range
        self.register_buffer("unit_xyz", torch.tensor(unit_xyz, dtype=torch.float), False)
        self.get_xyz = partial(
            cartesian, pc_range=pc_range, use_sigmoid=(xyz_activation=="sigmoid"))
        self.reverse_xyz = partial(
            reverse_cartesian, pc_range=pc_range, use_sigmoid=(xyz_activation=="sigmoid"))
        
        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            nn.Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim))

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
    ):
        output = self.layers(instance_feature + anchor_embed)

        #### for xyz
        delta_xyz = (2 * safe_sigmoid(output[..., :3]) - 1.) * self.unit_xyz[None, None]
        original_xyz = self.get_xyz(anchor[..., :3])
        anchor_xyz = original_xyz + delta_xyz
        anchor_xyz = self.reverse_xyz(anchor_xyz)

        #### for scale
        anchor_scale = output[..., 3:6]

        #### for rotation
        anchor_rotation = output[..., 6:10]
        anchor_rotation = torch.nn.functional.normalize(anchor_rotation, 2, -1)

        #### for opacity
        anchor_opa = output[..., 10:(10 + int(self.include_opa))]

        #### for semantic
        anchor_sem = output[..., self.semantic_start:(self.semantic_start + self.semantic_dim)]

        output = torch.cat([
            anchor_xyz, anchor_scale, anchor_rotation, anchor_opa, anchor_sem], dim=-1)
        
        xyz = self.get_xyz(anchor_xyz)

        if self.scale_act == 'sigmoid':
            scale = safe_sigmoid(anchor_scale)
        scale = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * scale
        
        if self.semantics_activation == 'softmax':
            semantics = anchor_sem.softmax(dim=-1)
        elif self.semantics_activation == 'softplus':
            semantics = F.softplus(anchor_sem)
        else:
            semantics = anchor_sem
        
        gaussian = GaussianPrediction(
            means=xyz,
            scales=scale,
            rotations=anchor_rotation,
            opacities=safe_sigmoid(anchor_opa),
            semantics=semantics,
            original_means=original_xyz,
            delta_means=delta_xyz
        )
        return output, gaussian #, semantics

    # def get_gaussian(self, output):
    #     if self.phi_activation == 'sigmoid':
    #         xyz = safe_sigmoid(output[..., :3])
    #     elif self.phi_activation == 'loop':
    #         xy = safe_sigmoid(output[..., :2])
    #         z = torch.remainder(output[..., 2:3], 1.0)
    #         xyz = torch.cat([xy, z], dim=-1)
    #     else:
    #         raise NotImplementedError
        
    #     if self.xyz_coordinate == 'polar':
    #         rrr = xyz[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
    #         theta = xyz[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
    #         phi = xyz[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
    #         xxx = rrr * torch.sin(theta) * torch.cos(phi)
    #         yyy = rrr * torch.sin(theta) * torch.sin(phi)
    #         zzz = rrr * torch.cos(theta)
    #     else:
    #         xxx = xyz[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
    #         yyy = xyz[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
    #         zzz = xyz[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
    #     xyz = torch.stack([xxx, yyy, zzz], dim=-1)

    #     gs_scales = safe_sigmoid(output[..., 3:6])
    #     gs_scales = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * gs_scales
        
    #     semantics = output[..., self.semantic_start: (self.semantic_start + self.semantic_dim)]
    #     if self.semantics_activation == 'softmax':
    #         semantics = semantics.softmax(dim=-1)
    #     elif self.semantics_activation == 'softplus':
    #         semantics = F.softplus(semantics)
        
    #     gaussian = GaussianPrediction(
    #         means=xyz,
    #         scales=gs_scales,
    #         rotations=output[..., 6:10],
    #         opacities=safe_sigmoid(output[..., 10: (10 + int(self.include_opa))]),
    #         semantics=semantics
    #     )
    #     return gaussian
