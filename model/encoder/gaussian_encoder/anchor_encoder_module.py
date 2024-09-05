from mmengine.registry import MODELS
from mmengine.model import BaseModule
from .utils import linear_relu_ln
import torch.nn as nn, torch


@MODELS.register_module()
class SparseGaussian3DEncoder(BaseModule):
    def __init__(
        self, 
        embed_dims: int = 256, 
        include_opa=True,
        semantics=False,
        semantic_dim=None
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.include_opa = include_opa
        self.semantics = semantics

        def embedding_layer(input_dims):
            return nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, input_dims))

        self.xyz_fc = embedding_layer(3)
        self.scale_fc = embedding_layer(3)
        self.rot_fc = embedding_layer(4)
        if include_opa:
            self.opacity_fc = embedding_layer(1)
        if semantics:
            assert semantic_dim is not None
            self.semantics_fc = embedding_layer(semantic_dim)
            self.semantic_start = 10 + int(include_opa)
        else:
            semantic_dim = 0
        self.semantic_dim = semantic_dim            
        self.output_fc = embedding_layer(self.embed_dims)

    def forward(self, box_3d: torch.Tensor):
        xyz_feat = self.xyz_fc(box_3d[..., :3])
        scale_feat = self.scale_fc(box_3d[..., 3:6])
        rot_feat = self.rot_fc(box_3d[..., 6:10])
        if self.include_opa:
            opacity_feat = self.opacity_fc(box_3d[..., 10:11])
        else:
            opacity_feat = 0.
        if self.semantics:
            semantic_feat = self.semantics_fc(box_3d[..., self.semantic_start: (self.semantic_start + self.semantic_dim)])
        else:
            semantic_feat = 0.

        output = xyz_feat + scale_feat + rot_feat + opacity_feat + semantic_feat
        output = self.output_fc(output)
        return output
