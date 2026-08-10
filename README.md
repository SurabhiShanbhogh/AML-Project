# Training Deep Learning Models for Industrial Waste Classification

**Module:** EEEM068 Applied Machine Learning | **Author:** Surabhi Shanbhogh

## Overview
Classifies cropped RGB waste images from **WaRP-C** into **28 fine-grained categories**. 

## Dataset
WaRP-C: 7,058 train / 1,765 val / 1,551 test images, 28 classes. Preprocessed once via `AML_WarpC_Processing_28Classes.ipynb` (pad to square, resize to 224x224, split) and reused identically across all four models, the preprocessed zip is downloaded and re-uploaded into each model notebook.

## Models & Results
DINOv2 and NFNet figures below are from each model's best-performing ablation config (see Ablations section).

| Model | Accuracy | F1 |
|---|---|---|
| RegNetY-400MF (baseline) | 62.54% | 61.10% |
| ConvNeXt-Tiny | 78.21% | 78.29% |
| **DINOv2 (best)** | **83.17%** | **83.02%** |
| NFNet | 79.30% | 79.10% |

DINOv2 (self-supervised ViT) performed best overall.

## Ablations
- **Frozen vs. fine-tuned:** fine-tuning essential for both DINOv2 (37.65pp gap) and NFNet (7.86pp gap).
- **Loss function:** unweighted/plain loss narrowly beat class-weighted loss for both new models. NFNet's Focal Loss run collapsed (51.84%).

## Quantization (INT8)
ConvNeXt-Tiny: near-lossless (79.60% → 79.83%). DINOv2: severe drop (83.17% → 70.34%) despite 3.78x size reduction.

## Key Finding
Confusion-matrix and Grad-CAM analysis show the same failure modes (`bottle-transp` variants, `juice-cardboard`/`milk-cardboard` confusion) across all four models , evidence the difficulty is in the data, not any one model.

## Notebooks
Preprocessing → per-model training (baseline, ConvNeXt-Tiny, DINOv2, NFNet + ablations) → `Cross_Model_Comparison.ipynb` → `Multi_Model_GradCAM_Comparison.ipynb` → `WaRP_S_Segmentation.ipynb`.
