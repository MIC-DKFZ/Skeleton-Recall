from typing import List, Union

import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.morphology import skeletonize, dilation

from batchgeneratorsv2.transforms.base.basic_transform import SegOnlyTransform


class SkeletonTransform(SegOnlyTransform):
    def __init__(self, do_tube: bool = True):
        """
        Calculates the skeleton of the segmentation (plus an optional 2 px tube around it)
        and appends it to the segmentation as an additional channel
        """
        super().__init__()
        self.do_tube = do_tube

    @staticmethod
    def get_tube_footprint(ndim: int) -> np.ndarray:
        # equivalent to dilating twice with the default footprint
        return ndi.iterate_structure(ndi.generate_binary_structure(ndim, 1), 2)

    def _apply_to_segmentation(self, segmentation: Union[torch.Tensor, List[torch.Tensor]], **params):
        if isinstance(segmentation, list):
            return [self._add_skeleton(segmentation[0])] + segmentation[1:]
        return self._add_skeleton(segmentation)

    def _add_skeleton(self, segmentation: torch.Tensor) -> torch.Tensor:
        seg = segmentation[0].numpy()
        # Add tubed skeleton GT
        bin_seg = seg > 0
        seg_skel = np.zeros_like(seg, dtype=np.int16)

        # Skeletonize
        if np.any(bin_seg):
            skel = skeletonize(bin_seg)
            skel = (skel > 0).astype(np.int16)
            if self.do_tube:
                skel = dilation(skel, footprint=self.get_tube_footprint(skel.ndim))
            seg_skel = skel * seg.astype(np.int16)

        return torch.cat((segmentation, torch.from_numpy(seg_skel)[None].to(segmentation.dtype)))
