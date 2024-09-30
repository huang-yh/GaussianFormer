import numpy as np
import torch, torch.nn as nn

from mmengine.registry import MODELS
from .base_head import BaseTaskHead
from .localagg.local_aggregate import LocalAggregator
from ..utils.utils import list_2_tensor, get_rotation_matrix


@MODELS.register_module()
class GaussianHead(BaseTaskHead):
    def __init__(
        self, 
        init_cfg=None,
        apply_loss_type=None,
        num_classes=18,
        empty_args=None,
        with_empty=False,
        cuda_kwargs=None,
        dataset_type='nusc',
        empty_label=17,
        **kwargs,
    ):
        super().__init__(init_cfg)
        
        self.num_classes = num_classes
        self.aggregator = LocalAggregator(**cuda_kwargs)
        if with_empty:
            self.empty_scalar = nn.Parameter(torch.ones(1, dtype=torch.float))
            self.register_buffer('empty_mean', torch.tensor(empty_args['mean'])[None, None, :])
            self.register_buffer('empty_scale', torch.tensor(empty_args['scale'])[None, None, :])
            self.register_buffer('empty_rot', torch.tensor([1., 0., 0., 0.])[None, None, :])
            self.register_buffer('empty_sem', torch.zeros(self.num_classes)[None, None, :])
            self.register_buffer('empty_opa', torch.ones(1)[None, None, :])
        self.with_emtpy = with_empty
        self.empty_args = empty_args
        self.dataset_type = dataset_type
        self.empty_label = empty_label

        if apply_loss_type == 'all':
            self.apply_loss_type = 'all'
        elif 'random' in apply_loss_type:
            self.apply_loss_type = 'random'
            self.random_apply_loss_layers = int(apply_loss_type.split('_')[1])
        else:
            raise NotImplementedError
        self.register_buffer('zero_tensor', torch.zeros(1, dtype=torch.float))

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def _sampling(self, gt_xyz, gt_label, gt_mask=None):
        if gt_mask is None:
            gt_label = gt_label.flatten(1)
            gt_xyz = gt_xyz.flatten(1, 3)
        else:
            assert gt_label.shape[0] == 1, "OccLoss does not support bs > 1"
            gt_label = gt_label[gt_mask].reshape(1, -1)
            gt_xyz = gt_xyz[gt_mask].reshape(1, -1, 3)
        return gt_xyz, gt_label

    def prepare_gaussian_args(self, gaussians):
        means = gaussians.means # b, g, 3
        scales = gaussians.scales # b, g, 3
        rotations = gaussians.rotations # b, g, 4
        opacities = gaussians.semantics # b, g, c
        origi_opa = gaussians.opacities # b, g, 1
        if origi_opa.numel() == 0:
            origi_opa = torch.ones_like(opacities[..., :1], requires_grad=False)
        if self.with_emtpy:
            assert opacities.shape[-1] == self.num_classes - 1
            if 'kitti' in self.dataset_type:
                opacities = torch.cat([torch.zeros_like(opacities[..., :1]), opacities], dim=-1)
            else:
                opacities = torch.cat([opacities, torch.zeros_like(opacities[..., :1])], dim=-1)
            means = torch.cat([means, self.empty_mean], dim=1)
            scales = torch.cat([scales, self.empty_scale], dim=1)
            rotations = torch.cat([rotations, self.empty_rot], dim=1)
            empty_sem = self.empty_sem.clone()
            empty_sem[..., self.empty_label] += self.empty_scalar
            opacities = torch.cat([opacities, empty_sem], dim=1)
            origi_opa = torch.cat([origi_opa, self.empty_opa], dim=1)

        bs, g, _ = means.shape
        S = torch.zeros(bs, g, 3, 3, dtype=means.dtype, device=means.device)
        S[..., 0, 0] = scales[..., 0]
        S[..., 1, 1] = scales[..., 1]
        S[..., 2, 2] = scales[..., 2]
        R = get_rotation_matrix(rotations) # b, g, 3, 3
        M = torch.matmul(S, R)
        Cov = torch.matmul(M.transpose(-1, -2), M)
        CovInv = Cov.cpu().inverse().cuda() # b, g, 3, 3
        return means, origi_opa, opacities, scales, CovInv

    def forward(
        self,
        representation,
        metas=None,
        **kwargs
    ):
        num_decoder = len(representation)
        if not self.training:
            apply_loss_layers = [num_decoder - 1]
        elif self.apply_loss_type == "all":
            apply_loss_layers = list(range(num_decoder))
        elif self.apply_loss_type == "random":
            if self.random_apply_loss_layers > 1:
                apply_loss_layers = np.random.choice(num_decoder - 1, self.random_apply_loss_layers - 1, False)
                apply_loss_layers = apply_loss_layers.tolist() + [num_decoder - 1]
            else:
                apply_loss_layers = [num_decoder - 1]
        else:
            raise NotImplementedError

        prediction = []
        occ_xyz = metas['occ_xyz'].to(self.zero_tensor.device)
        occ_label = metas['occ_label'].to(self.zero_tensor.device)
        occ_cam_mask = metas['occ_cam_mask'].to(self.zero_tensor.device)
        # occ_xyz = list_2_tensor(metas, 'occ_xyz', self.zero_tensor)
        # occ_label = list_2_tensor(metas, 'occ_label', self.zero_tensor)
        # occ_cam_mask = list_2_tensor(metas, 'occ_cam_mask', self.zero_tensor)
        # sampled_xyz, sampled_label = self._sampling(occ_xyz, occ_label, occ_cam_mask)
        sampled_xyz, sampled_label = self._sampling(occ_xyz, occ_label, None)
        for idx in apply_loss_layers:
            gaussians = representation[idx]['gaussian']

            means, origi_opa, opacities, scales, CovInv = self.prepare_gaussian_args(gaussians)
            bs, g = means.shape[:2]

            semantics = self.aggregator(
                sampled_xyz.clone().float(), 
                means, 
                origi_opa.reshape(bs, g),
                opacities,
                scales,
                CovInv)[None].transpose(1, 2) # 1, c, n
            
            prediction.append(semantics)
        
        return {
            'pred_occ': prediction,
            'sampled_label': sampled_label,
            'sampled_xyz': sampled_xyz,
            'occ_mask': occ_cam_mask,
            'gaussian': representation[-1]['gaussian']
        }

