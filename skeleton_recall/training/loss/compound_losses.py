import torch
from torch import nn

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from skeleton_recall.training.loss.dice import SoftSkeletonRecallLoss


class DC_SkelREC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, soft_skelrec_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, weight_srec=1,
                 ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss):
        super(DC_SkelREC_and_CE_loss, self).__init__()
        self.weight_srec = weight_srec
        self.ignore_label = ignore_label

        self.dc_and_ce = DC_and_CE_loss(soft_dice_kwargs, ce_kwargs, weight_ce=weight_ce, weight_dice=weight_dice,
                                        ignore_label=ignore_label, dice_class=dice_class)
        self.srec = SoftSkeletonRecallLoss(apply_nonlin=softmax_helper_dim1, **soft_skelrec_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=2 (segmentation, skeleton) or c=1 (segmentation only, no skeleton recall)
        """
        target_seg = target[:, :1]
        target_skel = target[:, 1:]

        result = self.dc_and_ce(net_output, target_seg)

        if self.weight_srec != 0 and target.shape[1] == 2:
            if self.ignore_label is not None:
                mask = target_seg != self.ignore_label
                target_skel = torch.where(mask, target_skel, 0)
            else:
                mask = None
            result = result + self.weight_srec * self.srec(net_output, target_skel, loss_mask=mask)
        return result
