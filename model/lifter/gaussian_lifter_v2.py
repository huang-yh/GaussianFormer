import torch, torch.nn as nn, math, os
import numpy as np
from einops import rearrange
from mmseg.registry import MODELS
from .base_lifter import BaseLifter
from ..utils.safe_ops import safe_inverse_sigmoid
from ..utils.sampler import DistributionSampler

try:
    from pointops import farthest_point_sampling
except:
    print("farthest_point_sampling import error.")


@MODELS.register_module()
class GaussianLifterV2(BaseLifter):
    def __init__(
        self,
        num_anchor,
        embed_dims,
        anchor_grad=True,
        feat_grad=True,
        semantics=False,
        semantic_dim=None,
        include_opa=True,
        xyz_activation="sigmoid",
        scale_activation="sigmoid",

        num_samples=64,
        pc_range=[-50, -50, -5, 50, 50, 3],
        voxel_size=0.5,
        occ_resolution=[200, 200, 16],
        empty_label=17,
        anchors_per_pixel=1,
        random_sampling=True,
        projection_in=None,
        initializer=None,
        initializer_img_downsample=None,
        pretrained_path=None,
        deterministic=True,
        random_samples=0,
        **kwargs,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.xyz_act = xyz_activation
        self.scale_act = scale_activation
        self.include_opa = include_opa
        self.semantics = semantics
        self.semantic_dim = semantic_dim

        self.random_samples = random_samples
        if random_samples > 0:
            self.random_anchors = self.init_random_anchors()
                    
        scale = torch.ones(num_anchor, 3, dtype=torch.float) * 0.5
        if scale_activation == "sigmoid":
            scale = safe_inverse_sigmoid(scale)

        rots = torch.zeros(num_anchor, 4, dtype=torch.float)
        rots[:, 0] = 1

        if include_opa:
            opacity = safe_inverse_sigmoid(0.5 * torch.ones((num_anchor, 1), dtype=torch.float))
        else:
            opacity = torch.ones((num_anchor, 0), dtype=torch.float)

        if semantics:
            assert semantic_dim is not None
        else:
            semantic_dim = 0
        semantic = torch.randn(num_anchor, semantic_dim, dtype=torch.float)
        anchor = torch.cat([scale, rots, opacity, semantic], dim=-1)

        self.num_anchor = num_anchor
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.instance_feature = nn.Parameter(
            torch.zeros([num_anchor + random_samples, self.embed_dims]),
            requires_grad=feat_grad,
        )
        projection_in = embed_dims * 4 if projection_in is None else projection_in
        self.projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(projection_in, num_samples + 1),
        )
        self.sampler = DistributionSampler()
        self.num_samples = num_samples
        self.register_buffer("depth_bins", torch.linspace(
            1.0, 72.0, self.num_samples, dtype=torch.float), persistent=False)
        self.register_buffer("pc_start", torch.tensor(
            pc_range[:3], dtype=torch.float), persistent=False)
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.occ_resolution = occ_resolution
        self.empty_label = empty_label
        self.anchors_per_pixel = anchors_per_pixel
        self.random_sampling = random_sampling
        if initializer is not None:
            self.initialize_backbone = MODELS.build(initializer)
        else:
            self.initialize_backbone = None
        self.initializer_img_downsample = initializer_img_downsample
        
        self.pretrained_path = pretrained_path
        self.deterministic = deterministic
        if pretrained_path is not None:
            ckpt = torch.load(pretrained_path, map_location='cpu')
            ckpt = ckpt.get("state_dict", ckpt)
            if 'instance_feature' in ckpt:
                del ckpt['instance_feature']
            if 'anchor' in ckpt:
                del ckpt['anchor']
            print(self.load_state_dict(ckpt, strict=False))
            print("Gaussian Initializer Weight Loaded Successfully.")

    def init_random_anchors(self):
        num_anchor = self.random_samples

        xyz = torch.rand(num_anchor, 3, dtype=torch.float)
        if self.xyz_act == "sigmoid":
            xyz = safe_inverse_sigmoid(xyz)
        
        scale = torch.ones(num_anchor, 3, dtype=torch.float) * 0.5
        if self.scale_act == "sigmoid":
            scale = safe_inverse_sigmoid(scale)

        rots = torch.zeros(num_anchor, 4, dtype=torch.float)
        rots[:, 0] = 1

        if self.include_opa:
            opacity = safe_inverse_sigmoid(0.5 * torch.ones((num_anchor, 1), dtype=torch.float))
        else:
            opacity = torch.ones((num_anchor, 0), dtype=torch.float)

        if self.semantics:
            semantic_dim = self.semantic_dim
            assert semantic_dim is not None
        else:
            semantic_dim = 0
        semantic = torch.randn(num_anchor, semantic_dim, dtype=torch.float)
        anchor = torch.cat([xyz, scale, rots, opacity, semantic], dim=-1)
        anchor = nn.Parameter(anchor, True)
        return anchor

    def init_weights(self):
        if self.pretrained_path is not None:
            return
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def forward(self, metas, **kwargs):
        if self.initialize_backbone is not None:
            b, n = kwargs["imgs"].shape[:2]
            initialize_input = kwargs["imgs"].flatten(0, 1)
            if self.initializer_img_downsample is not None:
                initialize_input = nn.functional.interpolate(
                    initialize_input, scale_factor=self.initializer_img_downsample, 
                    mode='bilinear', align_corners=True)
            secondfpn_out = self.initialize_backbone(initialize_input)
            secondfpn_out = secondfpn_out.unflatten(0, (b, n))
        else:
            secondfpn_out = kwargs["secondfpn_out"]
        
        b, n, _, h, w = secondfpn_out.shape
        feature = rearrange(secondfpn_out, 'b n c h w -> b n h w c')
        logits = self.projection(feature) # b, n, h, w, d + 1

        projection_mat = metas["projection_mat"].inverse() # img2lidar
        u = (torch.arange(w, dtype=feature.dtype, device=feature.device) + 0.5) / w
        v = (torch.arange(h, dtype=feature.dtype, device=feature.device) + 0.5) / h
        uv = torch.stack([
            u[None, :].expand(h, w), v[:, None].expand(h, w)], dim=-1) # h, w, 2
        uv = uv[None, None].expand(b, n, h, w, 2) * metas['image_wh'][:, :, None, None] # b, n, h, w, 2
        uvd = uv.unsqueeze(4).expand(b, n, h, w, self.num_samples, 2)
        uvd1 = torch.cat([uvd, torch.ones_like(uvd)], dim=-1) # b, n, h, w, d, 4
        uvd1[..., :3] = uvd1[..., :3] * self.depth_bins.view(1, 1, 1, 1, -1, 1)
        anchor_pts = projection_mat[:, :, None, None, None] @ uvd1[..., None]
        anchor_pts = anchor_pts.squeeze(-1)[..., :3]
        if kwargs.get("benchmarking", False):
            anchor_gt = None
        else:
            oob_mask = (anchor_pts[..., 0] < self.pc_range[0]) | (anchor_pts[..., 0] >= self.pc_range[3]) | \
                       (anchor_pts[..., 1] < self.pc_range[1]) | (anchor_pts[..., 1] >= self.pc_range[4]) | \
                       (anchor_pts[..., 2] < self.pc_range[2]) | (anchor_pts[..., 2] >= self.pc_range[5])
            anchor_idx = (anchor_pts - self.pc_start.view(1, 1, 1, 1, 1, 3)) / self.voxel_size
            anchor_idx = anchor_idx.to(torch.int)
            anchor_idx[..., 0].clamp_(0, self.occ_resolution[0] - 1)
            anchor_idx[..., 1].clamp_(0, self.occ_resolution[1] - 1)
            anchor_idx[..., 2].clamp_(0, self.occ_resolution[2] - 1)

            occupancy = metas["occ_label"]
            valid_mask = metas["occ_cam_mask"]
            anchor_occ = torch.stack([occ[idx[..., 0], idx[..., 1], idx[..., 2]] for occ, idx in zip(occupancy, anchor_idx)])
            anchor_occ[oob_mask] = self.empty_label
            anchor_valid = torch.stack([occ[idx[..., 0], idx[..., 1], idx[..., 2]] for occ, idx in zip(valid_mask, anchor_idx)])
            anchor_valid[oob_mask] = False
            anchor_gt = (anchor_occ != self.empty_label) & anchor_valid
            anchor_gt = torch.cat([anchor_gt, ~torch.any(anchor_gt, dim=-1, keepdim=True)], dim=-1)
        
        pdfs = torch.softmax(logits, dim=-1)
        deterministic = getattr(self, 'deterministic', True)
        index, pdf_i = self.sampler.sample(pdfs, deterministic, self.anchors_per_pixel) # b, n, h, w, a
        disable_mask = (pdfs.argmax(dim=-1, keepdim=True) == self.num_samples).expand(
            -1, -1, -1, -1, self.anchors_per_pixel)
        # disable_mask = index == self.num_samples
        sampled_anchor = self.sampler.gather(index.clamp(max=(self.num_samples-1)), anchor_pts) # b, n, h, w, a, 3
        
        anchor_xyz = []
        for i in range(b):
            cur_sampled_anchor = sampled_anchor[i][~disable_mask[i]]
            cur_oob_mask = (cur_sampled_anchor[..., 0] < self.pc_range[0]) | (cur_sampled_anchor[..., 0] >= self.pc_range[3]) | \
                   (cur_sampled_anchor[..., 1] < self.pc_range[1]) | (cur_sampled_anchor[..., 1] >= self.pc_range[4]) | \
                   (cur_sampled_anchor[..., 2] < self.pc_range[2]) | (cur_sampled_anchor[..., 2] >= self.pc_range[5])
            scan = cur_sampled_anchor[~cur_oob_mask]
            
            if self.random_sampling:
                if scan.shape[0] < self.num_anchor:
                    multi = int(math.ceil(self.num_anchor * 1.0 / scan.shape[0])) - 1
                    scan_ = scan.repeat(multi, 1)
                    scan_ = scan_ + torch.randn_like(scan_) * 0.1
                    scan_ = scan_[np.random.choice(scan_.shape[0], self.num_anchor - scan.shape[0], False)]
                    scan_[:, 0].clamp_(self.pc_range[0], self.pc_range[3])
                    scan_[:, 1].clamp_(self.pc_range[1], self.pc_range[4])
                    scan_[:, 2].clamp_(self.pc_range[2], self.pc_range[5])
                    scan = torch.cat([scan, scan_], 0)
                else:
                    scan = scan[np.random.choice(scan.shape[0], self.num_anchor, False)]
            else:
                if scan.shape[0] < self.num_anchor:
                    multi = int(math.ceil(self.num_anchor * 1.0 / scan.shape[0])) - 1
                    scan_ = scan.repeat(multi, 1)
                    scan_ = scan_ + torch.randn_like(scan_) * 0.1
                    scan_[:, 0].clamp_(self.pc_range[0], self.pc_range[3])
                    scan_[:, 1].clamp_(self.pc_range[1], self.pc_range[4])
                    scan_[:, 2].clamp_(self.pc_range[2], self.pc_range[5])
                    scan = torch.cat([scan, scan_], 0)
                # breakpoint()
                if kwargs.get("benchmarking", False):
                    scan = scan[np.random.permutation(scan.shape[0])]
                    num_subsets = 3
                    sublens = torch.linspace(0, scan.shape[0], num_subsets + 1, dtype=torch.int, device=scan.device)[1:]
                    new_sublens = torch.linspace(0, self.num_anchor, num_subsets + 1, dtype=torch.int, device=scan.device)[1:]
                    scanidx = farthest_point_sampling(scan, sublens, new_sublens)
                else:
                    # breakpoint()
                    scanidx = farthest_point_sampling(
                        scan, 
                        torch.tensor([scan.shape[0]], device=scan.device, dtype=torch.int),
                        torch.tensor([self.num_anchor], device=scan.device, dtype=torch.int))
                scan = scan[scanidx, :]
            
            anchor_xyz.append(scan)

            if os.environ.get("DEBUG", 'false') == 'true':
                prefix = 'kitti-'
                #### save pred scan
                np.save(f'{prefix}pred_scan.npy', scan.detach().cpu().numpy())
                #### save gt scan
                np.save('gt_scan_occ.npy', anchor_occ.detach().cpu().numpy())
                np.save('gt_scan_pts.npy', anchor_pts.detach().cpu().numpy())
                #### save gt occupancy
                np.save('gt_occ.npy', metas['occ_label'].detach().cpu().numpy())
                np.save('gt_pts.npy', metas['occ_xyz'].detach().cpu().numpy())

                #### obtain depth
                # occ_depth = anchor_gt.float().argmax(dim=-1) # b, n, h, w
                # oob_mask = occ_depth == 128
                # occ_depth = occ_depth.clamp_max(127)
                
                # occ_from_occ_depth = torch.gather(
                #     anchor_idx, -2, occ_depth[..., None, None].expand(-1, -1, -1, -1, -1, 3))
                # occ_from_occ_depth = occ_from_occ_depth[~oob_mask].reshape(-1, 3)
                # pred_occ = torch.zeros_like(occupancy[0], dtype=torch.bool)
                # pred_occ[
                #     occ_from_occ_depth[:, 0], 
                #     occ_from_occ_depth[:, 1], 
                #     occ_from_occ_depth[:, 2]] = True
                
                # occ = occupancy[i]
                # scan_idx = ((scan - self.pc_start.view(1, 3)) / self.voxel_size).int()
                # pred_occ = torch.zeros_like(occ, dtype=torch.bool)
                # pred_occ[scan_idx[..., 0], scan_idx[..., 1], scan_idx[..., 2]] = True
                # gt_occ = (occ != self.empty_label).bool()
                # correct = (pred_occ & gt_occ).sum()
                # recall = correct / gt_occ.sum()
                # print(f"recall: {recall}")
                # miou = correct / (gt_occ.sum() + pred_occ.sum() - correct)
                # print(f"miou: {miou}")
                # precision = correct / pred_occ.sum()
                # print(f"precision: {precision}")
                # # scan_occ = occ[scan_idx[..., 0], scan_idx[..., 1], scan_idx[..., 2]]
                # # precision = (scan_occ != self.empty_label).sum() / scan_occ.numel()
                # # print(f"precision: {precision}")
                breakpoint()
        
        anchor_xyz = torch.stack(anchor_xyz)
        anchor_xyz[..., 0] = (anchor_xyz[..., 0] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        anchor_xyz[..., 1] = (anchor_xyz[..., 1] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        anchor_xyz[..., 2] = (anchor_xyz[..., 2] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])

        if self.xyz_act == "sigmoid":
            xyz = safe_inverse_sigmoid(anchor_xyz)
        anchor = torch.cat([
            xyz, torch.tile(self.anchor[None], (b, 1, 1))], dim=-1)
        
        if self.random_samples > 0:
            random_anchors = torch.tile(self.random_anchors[None], (b, 1, 1))
            anchor = torch.cat([anchor, random_anchors], dim=1)

        instance_feature = torch.tile(
            self.instance_feature[None], (b, 1, 1)
        )
        return {
            'rep_features': instance_feature,
            'representation': anchor,
            'anchor_init': anchor[0].clone(),
            'pixel_logits': logits,
            'pixel_gt': anchor_gt,
        }