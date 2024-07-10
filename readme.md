# [ECCV 2024] Skeleton Recall Loss for Connectivity Conserving and Resource Efficient Segmentation of Thin Tubular Structures 🩻

## Overview
Welcome to the repository for the paper **"Skeleton Recall Loss for Connectivity Conserving and Resource Efficient Segmentation of Thin Tubular Structures"**! 🎉 This repository provides the code for the implementation of the Skeleton Recall Loss integrated within the popular nnUNet framework.
[![arXiv](https://img.shields.io/badge/arXiv-2404.03010-B31B1B.svg)](https://arxiv.org/abs/2404.03010)


## News/Updates:

- 📄 **7/24**: ECCV Acceptance
- 🪧 **12/23**: MedNeurIPS poster
- 🥇 **9/23**: Top solution without additional data on the [ToothFairy challenge](https://toothfairy.grand-challenge.org/)
- 🥇 **7/23**: Skeleton Recall won the [TopCoW challenge](https://topcow23.grand-challenge.org/)

## Introduction
Accurately segmenting thin tubular structures, such as vessels, nerves, roads, or cracks, is a crucial task in computer vision. Traditional deep learning-based segmentation approaches often struggle to preserve the connectivity of these structures. This paper introduces **Skeleton Recall Loss**, a novel loss function designed to enhance connectivity conservation in thin tubular structure segmentation without incurring massive computational overheads.

## Key Features
- **Connectivity Preservation**: Ensures the structural integrity of thin tubular structures in segmentation outputs.
- **Resource Efficiency**: Avoids the high computational cost associated with differentiable skeletonization methods.
- **Versatility**: Applicable to both 2D and 3D datasets, as well as binary and multi-class segmentation tasks.

## Methodology
The Skeleton Recall Loss operates by performing a tubed skeletonization on the ground truth segmentation and then computing a soft recall loss against the predicted segmentation output. This circumvents the costly calculation of a differentiable skeleton.

### Tubed Skeletonization
1. **Binarization**: Convert the ground truth segmentation mask to binary form.
2. **Skeleton Extraction**: Compute the skeleton using efficient methods for 2D and 3D inputs.
3. **Tubular Dilation**: Enlarge the skeleton using a dilation process to create a tubed skeleton.
4. **Class Assignment**: For multi-class problems, assign parts of the skeleton to their respective classes.

In the code the Tubed Skeletonization is done during *dataloading*, see the [code](nnunetv2/training/data_augmentation/custom_transforms/skeletonization.py).

### Soft Recall Loss
- **Soft Recall Calculation**: Compute the soft recall of the prediction on the precomputed tubed skeleton of the ground truth, see the [code](nnunetv2/training/loss/dice.py).
- **Combination with Generic Loss**: Combine with other generic loss functions (e.g., Dice Loss, Cross Entropy Loss) to enhance segmentation performance,  see the [code](nnunetv2/training/loss/compound_losses.py).

#### Full Loss calculation:

```math
\mathcal{L} = \mathcal{L}_{Dice} + \mathcal{L}_{CE} + w \cdot \mathcal{L}_{SkelRecall}
```

You can change the weight of the additional Skeleton Recall Loss term by modifying the value of  `self.weight_srec`  in the [nnUNetTrainerSkeletonRecall](nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainerSkeletonRecall.py)

## Experimental Setup
The method is validated on several public datasets featuring thin structures, including:
- [**Roads**](https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset): Aerial images of roads.
- [**DRIVE**](https://drive.grand-challenge.org/): Retinal blood vessels.
- [**Cracks**](https://zenodo.org/records/8215100): Concrete structure cracks.
- [**ToothFairy**](https://toothfairy.grand-challenge.org/): Inferior Alveolar Canal in 3D.
- [**TopCoW**](https://topcow23.grand-challenge.org/): Circle of Willis vessels in the brain.

## Usage
### Installation

Check out the official [nnUNet installation instructions](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md)

**TL;DR**

Clone the repository and install the required dependencies:
```bash
git clone https://github.com/MIC-DKFZ/skeleton-recall.git
cd skeleton-recall
pip install -e .
```
nnU-Net needs to know where you intend to save raw data, preprocessed data and trained models. For this you need to set a few environment variables. Please follow the instructions [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md).


### Training

To train a model using Skeleton Recall Loss with nnUNet, run:

for 2D:
```bash
nnUNetv2_train DATASET_NAME_OR_ID 2d FOLD -tr nnUNetTrainerSkeletonRecall
```

for 3D:
```bash
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD -tr nnUNetTrainerSkeletonRecall
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{kirchhoff2024skeleton,
  title={Skeleton Recall Loss for Connectivity Conserving and Resource Efficient Segmentation of Thin Tubular Structures},
  author={Kirchhoff, Yannick and Rokuss, Maximilian and Roy, Saikat and others},
  journal={European Conference on Computer Vision},
  year={2024}
}
```

Happy coding! 🚀

# Acknowledgements
<img src="documentation/assets/HI_Logo.png" height="100px" />

<img src="documentation/assets/dkfz_logo.png" height="100px" />

nnU-Net is developed and maintained by the Applied Computer Vision Lab (ACVL) of [Helmholtz Imaging](http://helmholtz-imaging.de) 
and the [Division of Medical Image Computing](https://www.dkfz.de/en/mic/index.php) at the 
[German Cancer Research Center (DKFZ)](https://www.dkfz.de/en/index.html).