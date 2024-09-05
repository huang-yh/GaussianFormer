from mmseg.models import SEGMENTORS, builder
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule


@SEGMENTORS.register_module()
class CustomBaseSegmentor(BaseModule):

    def __init__(
        self,
        img_backbone=None,
        img_neck=None,
        lifter=None,
        encoder=None,
        head=None, 
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        if img_backbone is not None:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            try:
                self.img_neck = builder.build_neck(img_neck)
            except:
                self.img_neck = MODELS.build(img_neck)
        if lifter is not None:
            self.lifter = builder.build_head(lifter)
        if encoder is not None:
            self.encoder = builder.build_head(encoder)
        if head is not None:
            self.head = builder.build_head(head)

    def extract_img_feat(self, imgs, **kwargs):
        """Extract features of images."""
        B = imgs.size(0)

        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats = self.img_backbone(imgs)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
        img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return {'ms_img_feats': img_feats_reshaped}

    def forward(
        self,
        imgs,
        metas,
        **kwargs
    ):
        pass