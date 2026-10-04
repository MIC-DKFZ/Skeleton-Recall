import warnings

import torch

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from skeleton_recall.training.data_augmentation.skeletonization import SkeletonTransform
from skeleton_recall.training.loss.compound_losses import DC_SkelREC_and_CE_loss


def add_skeleton_transform(transforms: BasicTransform, skeleton_deep_supervision: bool,
                           do_tube: bool = True) -> BasicTransform:
    assert isinstance(transforms, ComposeTransforms)
    ds_idx = next((i for i, t in enumerate(transforms.transforms) if isinstance(t, DownsampleSegForDSTransform)),
                  None)
    if ds_idx is None:
        idx = len(transforms.transforms)
    else:
        idx = ds_idx if skeleton_deep_supervision else ds_idx + 1
    transforms.transforms.insert(idx, SkeletonTransform(do_tube=do_tube))
    return transforms


def strip_skeleton(target):
    if isinstance(target, list):
        return [i[:, :1] for i in target]
    return target[:, :1]


class nnUNetTrainerSkeletonRecall(nnUNetTrainer):
    skeleton_deep_supervision = False

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.weight_srec = 1  # This is the default value, you can change it if you want
        if self.label_manager.has_regions:
            raise NotImplementedError("trainer not implemented for regions")

    def _build_loss(self):
        if self.label_manager.ignore_label is not None:
            warnings.warn('Support for ignore label with Skeleton Recall is experimental and may not work as expected')
        loss = DC_SkelREC_and_CE_loss(soft_dice_kwargs={'batch_dice': self.configuration_manager.batch_dice,
                                                        'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                                      soft_skelrec_kwargs={'batch_dice': self.configuration_manager.batch_dice,
                                                           'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                                      ce_kwargs={}, weight_ce=1, weight_dice=1, weight_srec=self.weight_srec,
                                      ignore_label=self.label_manager.ignore_label,
                                      dice_class=MemoryEfficientSoftDiceLoss)

        if self._do_i_compile():
            loss.dc_and_ce.dc = torch.compile(loss.dc_and_ce.dc)

        default_loss = super()._build_loss()
        if isinstance(default_loss, DeepSupervisionWrapper):
            default_loss.loss = loss
            return default_loss
        return loss

    def get_training_transforms(self, *args, **kwargs) -> BasicTransform:
        return add_skeleton_transform(super().get_training_transforms(*args, **kwargs),
                                      self.skeleton_deep_supervision)

    def get_validation_transforms(self, *args, **kwargs) -> BasicTransform:
        return add_skeleton_transform(super().get_validation_transforms(*args, **kwargs),
                                      self.skeleton_deep_supervision)

    def validation_step(self, batch: dict) -> dict:
        # the loss needs the skeleton, the online evaluation in nnUNetTrainer.validation_step must not see it
        target = batch['target']
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        loss = self.loss
        self.loss = lambda output, _: loss(output, target)
        try:
            return super().validation_step({**batch, 'target': strip_skeleton(target)})
        finally:
            self.loss = loss


class nnUNetTrainerSkeletonRecallDS(nnUNetTrainerSkeletonRecall):
    skeleton_deep_supervision = True
