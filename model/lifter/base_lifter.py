from mmseg.registry import MODELS
from mmengine.model import BaseModule


@MODELS.register_module()
class BaseLifter(BaseModule):

    """Base lifter class.
    image backbone -> neck -> lifter -> encoder -> segmentor
    Lift multi-scale image features to 3D representations, e.g. Voxels or TPV or BEV.
    """

    def __init__(self, init_cfg=None, **kwargs) -> None:
        super().__init__(init_cfg)
    
    def forward(
        self, 
        ms_img_feats, 
        metas=None, 
        **kwargs
    ):
        pass