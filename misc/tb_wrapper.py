from torch.utils.tensorboard import SummaryWriter
from mmengine.utils import ManagerMixin

class WrappedTBWriter(SummaryWriter, ManagerMixin):

    def __init__(self, name, **kwargs):
        SummaryWriter.__init__(self, **kwargs)
        ManagerMixin.__init__(self, name)
