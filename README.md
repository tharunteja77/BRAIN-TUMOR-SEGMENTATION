# BRAIN-TUMOR-SEGMENTATION

# Brain Tumor Segmentation Using U-Net

This repository contains a deep learning project focused on segmenting brain tumors from MRI images using the U-Net convolutional neural network (CNN) architecture. The model is trained on the TCGA-LGG dataset obtained from The Cancer Imaging Archive (TCIA) and is capable of generating accurate binary masks that delineate tumor regions in FLAIR MRI scans.

## 🧠 Project Overview

Lower-grade gliomas (LGGs) are a heterogeneous group of brain tumors, and their accurate segmentation is crucial for diagnosis, treatment planning, and monitoring. This project automates the segmentation process using a U-Net model, which is especially well-suited for biomedical image segmentation tasks even with limited data.

- **Dataset:** 110 patient cases with FLAIR MRI images and corresponding ground truth masks.
- **Architecture:** U-Net (original implementation)
- **Loss Function:** Dice Loss
- **Optimizer:** Adamax
- **Metrics Used:** Dice Coefficient, Intersection over Union (IoU), Accuracy
- **Best Results:**
  - Dice Coefficient: 0.913
  - IoU: 0.841
  - Accuracy: 0.998

## 📁 Files

- `Brain_Tumor_Segmentation_Unet.ipynb`: Jupyter notebook with the full model pipeline, including preprocessing, training, evaluation, and visualization.
- `emt_project_report.pdf`: Final report containing detailed documentation, background literature, methodology, and results.

## 🏗️ Methodology

1. **Data Preprocessing**:
   - Normalization and resizing of FLAIR MRI images
   - Data augmentation: rotation, flipping, scaling

2. **Model Architecture**:
   - Contracting path with convolution + ReLU + max-pooling
   - Expansive path with transposed convolution and skip connections
   - Sigmoid output for binary segmentation mask

3. **Training Details**:
   - Optimizer: Adamax (lr=0.001)
   - Loss Function: Dice Loss
   - Epochs: 119 (early stopping)

4. **Evaluation**:
   - Dice Coefficient to assess overlap with ground truth
   - IoU to measure predictive precision
   - Accuracy to evaluate overall correctness

## 📊 Results

The model achieved high segmentation performance on the validation dataset with excellent generalization and minimal false positives.

| Metric           | Value  |
|------------------|--------|
| Dice Coefficient | 0.913  |
| IoU              | 0.841  |
| Accuracy         | 0.998  |

## 📚 References

- Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015.
- The Cancer Imaging Archive (TCIA) - TCGA-LGG Dataset
- M. Buda et al., 2019; M.A. Mazurowski et al., 2017

## 🙏 Acknowledgments

We thank:
- **TCIA and TCGA** for the dataset
- **U-Net authors** for the foundational architecture
- Our project guide and faculty for their continuous support



---

