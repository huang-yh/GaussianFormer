
import numpy as np
from mmengine import MMLogger
logger = MMLogger.get_instance('selfocc')
import torch.distributed as dist
import torch


class MeanIoU:

    def __init__(self,
                 class_indices,
                #  ignore_label: int,
                 empty_label,
                 label_str,
                 use_mask=False,
                 dataset_empty_label=17,
                 filter_minmax=True,
                 name = 'none'):
        self.class_indices = class_indices
        self.num_classes = len(class_indices)
        # self.ignore_label = ignore_label
        self.empty_label = empty_label
        self.dataset_empty_label = dataset_empty_label
        self.label_str = label_str
        self.use_mask = use_mask
        self.filter_minmax = filter_minmax
        self.name = name

    def reset(self) -> None:
        self.total_seen = torch.zeros(self.num_classes+1).cuda()
        self.total_correct = torch.zeros(self.num_classes+1).cuda()
        self.total_positive = torch.zeros(self.num_classes+1).cuda()

    def _after_step(self, outputs, targets, mask=None):
        # outputs = outputs[targets != self.ignore_label]
        # targets = targets[targets != self.ignore_label]
        if not isinstance(targets, (torch.Tensor, np.ndarray)):
            assert mask is None
            labels = torch.from_numpy(targets['semantics']).cuda()
            masks = torch.from_numpy(targets['mask_camera']).bool().cuda()
            targets = labels
            targets[targets == self.dataset_empty_label] = self.empty_label
            if self.filter_minmax:
                max_z = (targets != self.empty_label).nonzero()[:, 2].max()
                min_z = (targets != self.empty_label).nonzero()[:, 2].min()
                outputs[..., (max_z + 1):] = self.empty_label
                outputs[..., :min_z] = self.empty_label
            if self.use_mask:
                outputs = outputs[masks]
                targets = targets[masks]
        else:
            if mask is not None:
                outputs = outputs[mask]
                targets = targets[mask]

        for i, c in enumerate(self.class_indices):
            self.total_seen[i] += torch.sum(targets == c).item()
            self.total_correct[i] += torch.sum((targets == c)
                                               & (outputs == c)).item()
            self.total_positive[i] += torch.sum(outputs == c).item()
        
        self.total_seen[-1] += torch.sum(targets != self.empty_label).item()
        self.total_correct[-1] += torch.sum((targets != self.empty_label)
                                            & (outputs != self.empty_label)).item()
        self.total_positive[-1] += torch.sum(outputs != self.empty_label).item()

    def _after_epoch(self):
        if dist.is_initialized():
            dist.all_reduce(self.total_seen)
            dist.all_reduce(self.total_correct)
            dist.all_reduce(self.total_positive)
            dist.barrier()

        ious = []
        precs = []
        recas = []

        for i in range(self.num_classes):
            if self.total_positive[i] == 0:
                precs.append(0.)
            else:
                cur_prec = self.total_correct[i] / self.total_positive[i]
                precs.append(cur_prec.item())
            if self.total_seen[i] == 0:
                ious.append(1)
                recas.append(1)
            else:
                cur_iou = self.total_correct[i] / (self.total_seen[i]
                                                   + self.total_positive[i]
                                                   - self.total_correct[i])
                cur_reca = self.total_correct[i] / self.total_seen[i]
                ious.append(cur_iou.item())
                recas.append(cur_reca)

        miou = np.mean(ious)
        # logger = get_root_logger()
        logger.info(f'Validation per class iou {self.name}:')
        for iou, prec, reca, label_str in zip(ious, precs, recas, self.label_str):
            logger.info('%s : %.2f%%, %.2f, %.2f' % (label_str, iou * 100, prec, reca))
        
        logger.info(self.total_seen.int())
        logger.info(self.total_correct.int())
        logger.info(self.total_positive.int())

        occ_iou = self.total_correct[-1] / (self.total_seen[-1]
                                            + self.total_positive[-1]
                                            - self.total_correct[-1])
        # logger.info(f'iou: {occ_iou}')
        
        return miou * 100, occ_iou * 100
