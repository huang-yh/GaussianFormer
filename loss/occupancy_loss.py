import torch, torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmseg.models.losses import DiceLoss

from . import OPENOCC_LOSS
from .base_loss import BaseLoss
from .utils.lovasz_softmax import lovasz_softmax


nusc_class_frequencies = np.array([
    944004,
    1897170,
    152386,
    2391677,
    16957802,
    724139,
    189027,
    2074468,
    413451,
    2384460,
    5916653,
    175883646,
    4275424,
    51393615,
    61411620,
    105975596,
    116424404,
    1892500630
])


@OPENOCC_LOSS.register_module()
class OccupancyLoss(BaseLoss):

    def __init__(
        self,
        weight=1.0,
        empty_label=17,
        num_classes=18,
        use_focal_loss=False,
        focal_loss_args=dict(),
        use_dice_loss=False,
        balance_cls_weight=False,
        multi_loss_weights=dict(),
        use_sem_geo_scal_loss=True,
        use_lovasz_loss=True,
        lovasz_ignore=255,
        manual_class_weight=None,
        ignore_empty=False,
        lovasz_use_softmax=True,
        input_dict=None
    ):
        
        super().__init__()

        self.weight = weight
        if input_dict is None:
            self.input_dict = {
                'pred_occ': 'pred_occ',
                'sampled_xyz': 'sampled_xyz',
                'sampled_label': 'sampled_label',
                'occ_mask': 'occ_mask'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.loss_voxel

        self.empty_label = empty_label
        self.num_classes = num_classes
        self.classes = list(range(num_classes))
        self.use_sem_geo_scal_loss = use_sem_geo_scal_loss
        self.use_lovasz_loss = use_lovasz_loss
        self.lovasz_ignore = lovasz_ignore
        self.ignore_empty = ignore_empty
        self.lovasz_use_softmax = lovasz_use_softmax

        self.loss_voxel_ce_weight = multi_loss_weights.get('loss_voxel_ce_weight', 1.0)
        self.loss_voxel_sem_scal_weight = multi_loss_weights.get('loss_voxel_sem_scal_weight', 1.0)
        self.loss_voxel_geo_scal_weight = multi_loss_weights.get('loss_voxel_geo_scal_weight', 1.0)
        self.loss_voxel_lovasz_weight = multi_loss_weights.get('loss_voxel_lovasz_weight', 1.0)
        if balance_cls_weight:
            if manual_class_weight is not None:
                self.class_weights = torch.tensor(manual_class_weight)
            else:
                class_freqs = nusc_class_frequencies
                self.class_weights = torch.from_numpy(1 / np.log(class_freqs[:num_classes] + 0.001))
            self.class_weights = num_classes * F.normalize(self.class_weights, 1, -1)
            print(self.__class__, self.class_weights)
        else:
            self.class_weights = torch.ones(num_classes)

        self.use_focal_loss = use_focal_loss
        if use_focal_loss:
            self.focal_loss = CustomFocalLoss(**focal_loss_args)
        self.use_dice_loss = use_dice_loss
        if use_dice_loss:
            self.dice_loss = DiceLoss(
                class_weight=self.class_weights,
                loss_weight=2.0)
        
    def loss_voxel(self, pred_occ, sampled_xyz, sampled_label, occ_mask=None):

        tot_loss = 0.
        if self.ignore_empty:
            empty_mask = sampled_label != self.empty_label
            occ_mask = empty_mask if occ_mask is None else empty_mask & occ_mask.flatten(1)

        if occ_mask is not None:
            occ_mask = occ_mask.flatten(1)
            sampled_label = sampled_label[occ_mask][None]

        for semantics in pred_occ:
            if occ_mask is not None:
                semantics = semantics.transpose(1, 2)[occ_mask][None].transpose(1, 2) # 1, c, n
            loss_dict = {}
            # semantics = semantics.transpose(0, 1).unsqueeze(0)
            if self.use_focal_loss:
                loss_dict['loss_voxel_ce'] = self.loss_voxel_ce_weight * \
                    self.focal_loss(semantics, sampled_label, sampled_xyz, self.class_weights.type_as(semantics), ignore_index=255)
            else:
                if self.lovasz_use_softmax:
                    loss_dict['loss_voxel_ce'] = self.loss_voxel_ce_weight * \
                        CE_ssc_loss(semantics, sampled_label, self.class_weights.type_as(semantics), ignore_index=255)
                else:
                    loss_dict['loss_voxel_ce'] = self.loss_voxel_ce_weight * CE_wo_softmax(
                        semantics, sampled_label, self.class_weights.type_as(semantics), ignore_index=255)
            if self.use_sem_geo_scal_loss:
                if self.lovasz_use_softmax:
                    scal_input = torch.softmax(semantics, dim=1)
                else:
                    scal_input = semantics
                loss_dict['loss_voxel_sem_scal'] = self.loss_voxel_sem_scal_weight * sem_scal_loss(scal_input.clone(), sampled_label, ignore_index=255)
                loss_dict['loss_voxel_geo_scal'] = self.loss_voxel_geo_scal_weight * geo_scal_loss(scal_input.clone(), sampled_label, ignore_index=255, non_empty_idx=self.empty_label)
            if self.use_lovasz_loss:
                if self.lovasz_use_softmax:
                    lovasz_input = torch.softmax(semantics, dim=1)
                else:
                    lovasz_input = semantics
                loss_dict['loss_voxel_lovasz'] = self.loss_voxel_lovasz_weight * lovasz_softmax(
                    lovasz_input.transpose(1, 2).flatten(0, 1), sampled_label.flatten(), ignore=self.lovasz_ignore)
            if self.use_dice_loss:
                loss_dict['loss_voxel_dice'] = self.dice_loss(semantics, sampled_label)

            loss = 0.
            for k, v in loss_dict.items():
                loss = loss + v
            tot_loss = tot_loss + loss
        return tot_loss / len(pred_occ)


from torch.cuda.amp import autocast

def inverse_sigmoid(x, sign='A'):
    x = x.to(torch.float32)
    while x >= 1-1e-5:
        x = x - 1e-5

    while x< 1e-5:
        x = x + 1e-5

    return -torch.log((1 / x) - 1)

def CE_ssc_loss(pred, target, class_weights=None, ignore_index=255):
    """
    :param: prediction: the predicted tensor, must be [BS, C, ...]
    """

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=ignore_index, reduction="mean"
    )
    # from IPython import embed
    # embed()
    # exit()
    with autocast(False):
        loss = criterion(pred, target.long())

    return loss

def CE_wo_softmax(pred, target, class_weights=None, ignore_index=255):
    pred = torch.clamp(pred, 1e-6, 1. - 1e-6)
    loss = F.nll_loss(torch.log(pred), target, class_weights, ignore_index=ignore_index)
    return loss

def sem_scal_loss(pred, ssc_target, ignore_index=255):
    # Get softmax probabilities
    with autocast(False):
        # pred = F.softmax(pred_, dim=1)
        loss = 0
        count = 0
        mask = ssc_target != ignore_index
        n_classes = pred.shape[1]
        begin = 1 if n_classes == 19 else 0
        for i in range(begin, n_classes-1):   

            # Get probability of class i
            p = pred[:, i]  

            # Remove unknown voxels
            target_ori = ssc_target
            p = p[mask]
            target = ssc_target[mask]   

            completion_target = torch.ones_like(target)
            completion_target[target != i] = 0
            completion_target_ori = torch.ones_like(target_ori).float()
            completion_target_ori[target_ori != i] = 0
            if torch.sum(completion_target) > 0:
                count += 1.0
                nominator = torch.sum(p * completion_target)
                loss_class = 0
                if torch.sum(p) > 0:
                    precision = nominator / (torch.sum(p)+ 1e-5)
                    loss_precision = F.binary_cross_entropy_with_logits(
                            inverse_sigmoid(precision, 'D'), torch.ones_like(precision)
                        )
                    loss_class += loss_precision
                if torch.sum(completion_target) > 0:
                    recall = nominator / (torch.sum(completion_target) +1e-5)
                    # loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))

                    loss_recall = F.binary_cross_entropy_with_logits(inverse_sigmoid(recall, 'E'), torch.ones_like(recall))
                    loss_class += loss_recall
                if torch.sum(1 - completion_target) > 0:
                    specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                        torch.sum(1 - completion_target) +  1e-5
                    )

                    loss_specificity = F.binary_cross_entropy_with_logits(
                            inverse_sigmoid(specificity, 'F'), torch.ones_like(specificity)
                        )
                    loss_class += loss_specificity
                loss += loss_class
                # print(i, loss_class, loss_recall, loss_specificity)
        l = loss/count
        if torch.isnan(l):
            from IPython import embed
            embed()
            exit()
        return l

def geo_scal_loss(pred, ssc_target, ignore_index=255, non_empty_idx=0):

    # Get softmax probabilities
    # pred = F.softmax(pred, dim=1)

    # Compute empty and nonempty probabilities
    empty_probs = pred[:, non_empty_idx]
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != ignore_index
    nonempty_target = ssc_target != non_empty_idx
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    eps = 1e-5
    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / (nonempty_probs.sum()+eps)
    recall = intersection / (nonempty_target.sum()+eps)
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / ((1 - nonempty_target).sum()+eps)
    with autocast(False):
        return (
            F.binary_cross_entropy_with_logits(inverse_sigmoid(precision, 'A'), torch.ones_like(precision))
            + F.binary_cross_entropy_with_logits(inverse_sigmoid(recall, 'B'), torch.ones_like(recall))
            + F.binary_cross_entropy_with_logits(inverse_sigmoid(spec, 'C'), torch.ones_like(spec))
        )

from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss
from mmcv.ops import softmax_focal_loss as _softmax_focal_loss
from mmdet.models.losses.utils import weight_reduce_loss

# This method is only for debugging
def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
        loss = loss * weight

    loss = loss.sum(-1).mean()
    # loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def py_focal_loss_with_prob(pred,
                            target,
                            weight=None,
                            gamma=2.0,
                            alpha=0.25,
                            reduction='mean',
                            avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.
    Different from `py_sigmoid_focal_loss`, this function accepts probability
    as input.
    Args:
        pred (torch.Tensor): The prediction probability with shape (N, C),
            C is the number of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    num_classes = pred.size(1)
    target = F.one_hot(target, num_classes=num_classes + 1)
    target = target[:, :num_classes]

    target = target.type_as(pred)
    pt = (1 - pred) * target + pred * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy(
        pred, target, reduction='none') * focal_weight

    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       cls_weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    r"""A wrapper of cuda version `Focal Loss
    <https://arxiv.org/abs/1708.02002>`_.
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(), gamma,
                               alpha, cls_weight, 'none')
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
        loss = loss * weight
    loss = loss.sum(-1).mean()
    # loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

def softmax_focal_loss(pred,
                       target,
                       weight=None,
                       cls_weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    r"""A wrapper of cuda version `Focal Loss
    <https://arxiv.org/abs/1708.02002>`_.
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    loss = _softmax_focal_loss(pred.contiguous(), target.contiguous(), gamma,
                               alpha, cls_weight, 'none')
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
        loss = loss * weight
    loss = loss.mean()
    # loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

class CustomFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0,
                 activated=False):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_
        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            activated (bool, optional): Whether the input is activated.
                If True, it means the input has been activated and can be
                treated as probabilities. Else, it should be treated as logits.
                Defaults to False.
        """
        super(CustomFocalLoss, self).__init__()
        # assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated  
        
    def forward(self,
                pred,
                target,
                target_xyz,
                weight=None,
                avg_factor=None,
                ignore_index=255,
                reduction_override=None):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        target_xy = target_xyz[..., :2] # b, n, 2
        target_dist = torch.norm(target_xy, 2, -1) # b, n
        target_dist_max = target_dist.max()
        c = target_dist / target_dist_max + 1 # b, n
        c = c.flatten()

        # weight_mask = weight[None, None, :] * c[..., None] # b, n, c
        # weight_mask = weight_mask.flatten(0, 1)

        pred = pred.transpose(1, 2).flatten(0, 1) # BN, C
        target = target.flatten(0, 1)

        num_classes = pred.size(1)
        
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            if self.activated:
                calculate_loss_func = py_focal_loss_with_prob
            else:
                if torch.cuda.is_available() and pred.is_cuda:
                    calculate_loss_func = sigmoid_focal_loss
                else:
                    assert False
                    num_classes = pred.size(1)
                    target = F.one_hot(target, num_classes=num_classes + 1)
                    target = target[:, :num_classes]
                    calculate_loss_func = py_sigmoid_focal_loss
        else:
            calculate_loss_func = softmax_focal_loss

        loss_cls = self.loss_weight * calculate_loss_func(
            pred,
            target.to(torch.long),
            c,
            weight,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=reduction,
            avg_factor=avg_factor)

        return loss_cls
