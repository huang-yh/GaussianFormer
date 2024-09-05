import torch
from torch.autograd.function import Function, once_differentiable

from . import deformable_aggregation_ext


class DeformableAggregationFunction(Function):
    @staticmethod
    def forward(
        ctx,
        mc_ms_feat,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    ):
        # output: [bs, num_pts, num_embeds]
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()
        output = deformable_aggregation_ext.deformable_aggregation_forward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        ctx.save_for_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        ) = ctx.saved_tensors
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()

        grad_mc_ms_feat = torch.zeros_like(mc_ms_feat)
        grad_sampling_location = torch.zeros_like(sampling_location)
        grad_weights = torch.zeros_like(weights)
        deformable_aggregation_ext.deformable_aggregation_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
            grad_output.contiguous(),
            grad_mc_ms_feat,
            grad_sampling_location,
            grad_weights,
        )
        return (
            grad_mc_ms_feat,
            None,
            None,
            grad_sampling_location,
            grad_weights,
        )

    @staticmethod
    def feature_maps_format(feature_maps, inverse=False):
        bs, num_cams = feature_maps[0].shape[:2]
        if not inverse:
            spatial_shape = []
            scale_start_index = [0]

            col_feats = []
            for i, feat in enumerate(feature_maps):
                spatial_shape.append(feat.shape[-2:])
                scale_start_index.append(
                    feat.shape[-1] * feat.shape[-2] + scale_start_index[-1]
                )
                col_feats.append(torch.reshape(
                    feat, (bs, num_cams, feat.shape[2], -1)
                ))
            scale_start_index.pop()
            col_feats = torch.cat(col_feats, dim=-1).permute(0, 1, 3, 2)
            feature_maps = [
                col_feats,
                torch.tensor(
                    spatial_shape,
                    dtype=torch.int64,
                    device=col_feats.device,
                ),
                torch.tensor(
                    scale_start_index,
                    dtype=torch.int64,
                    device=col_feats.device,
                ),
            ]
        else:
            spatial_shape = feature_maps[1].int()
            split_size = (spatial_shape[:, 0] * spatial_shape[:, 1]).tolist()
            feature_maps = feature_maps[0].permute(0, 1, 3, 2)
            feature_maps = list(torch.split(feature_maps, split_size, dim=-1))
            for i, feat in enumerate(feature_maps):
                feature_maps[i] = feat.reshape(
                    feat.shape[:3] + (spatial_shape[i, 0], spatial_shape[i, 1])
                )
        return feature_maps
