from mmengine.model import BaseModule
from mmengine.registry import MODELS
from mmseg.models import builder
from mmdet3d.registry import MODELS as mmdet3dMODELS
import torch


@MODELS.register_module()
class ResNetSecondFPN(BaseModule):
    def __init__(
        self, 
        img_backbone_config, 
        neck_confifg,
        img_backbone_out_indices,
        pretrained_path=None
    ):

        super().__init__()

        self.img_backbone = builder.build_backbone(img_backbone_config)
        self.img_neck = mmdet3dMODELS.build(neck_confifg)
        self.img_backbone_out_indices = img_backbone_out_indices
        if pretrained_path is not None:
            ckpt = torch.load(pretrained_path, map_location='cpu')
            ckpt = ckpt.get("state_dict", ckpt)
            print(self.load_state_dict(ckpt, strict=False))
            print("ResNetSecondFPN Weight Loaded Successfully.")

    def forward(self, imgs):
        img_feats_backbone = self.img_backbone(imgs)
        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())
        img_feats = []
        for idx in self.img_backbone_out_indices:
            img_feats.append(img_feats_backbone[idx])
        secondfpn_out = self.img_neck(img_feats)[0]
        return secondfpn_out
