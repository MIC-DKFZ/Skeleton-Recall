from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import \
    nnUNetTrainerNoMirroring, nnUNetTrainer_onlyMirror01
from skeleton_recall.training.nnUNetTrainer.nnUNetTrainerSkeletonRecall import nnUNetTrainerSkeletonRecall, \
    nnUNetTrainerSkeletonRecallDS


class nnUNetTrainerSkeletonRecallNoMirroring(nnUNetTrainerSkeletonRecall, nnUNetTrainerNoMirroring):
    pass


class nnUNetTrainerSkeletonRecall_onlyMirror01(nnUNetTrainerSkeletonRecall, nnUNetTrainer_onlyMirror01):
    pass


class nnUNetTrainerSkeletonRecallDSNoMirroring(nnUNetTrainerSkeletonRecallDS, nnUNetTrainerNoMirroring):
    pass


class nnUNetTrainerSkeletonRecallDS_onlyMirror01(nnUNetTrainerSkeletonRecallDS, nnUNetTrainer_onlyMirror01):
    pass
