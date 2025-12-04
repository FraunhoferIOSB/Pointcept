# GOOSE 3D Semantic Segmentation Challenge - 1st Place Solution

This document outlines the winning solution for the GOOSE 3D Semantic Segmentation Challenge. The approach utilizes a novel architecture that combines the **Point Transformer v3 (PTv3)** backbone with a **Point Prompt Tuning (PPT)** framework to effectively process diverse 3D point cloud data from multiple robotic platforms.

## Methodology Overview

The solution addresses the challenge of learning from heterogeneous data sources (different robots like ALICE, MuCAR-3, Spot) while maintaining a unified semantic understanding.

### Key Components
1.  **PTv3 Backbone**: Serves as the primary feature extractor, leveraging transformer-based architecture to capture complex local and global geometric relationships.
2.  **Point Prompt Framework (PPT)**:
    *   **Data-Driven Context Adaptation**: Incorporates conditional processing (e.g., conditional normalization) based on the specific dataset origin or platform (e.g., "car", "alice", "spot").
    *   **Cross-Dataset Class Alignment**:
        *   **Language-Driven Categorical Alignment**: Uses CLIP to bridge different label spaces.
        *   **Decoupled Alignment**: Employs separate segmentation heads for each data condition while sharing backbone features.

## Dataset Preparation

The dataset should be structured as follows:

```text
goose_sep
├── alice
│   ├── 3d_challenge
│   │   ├── train
│   │   └── val
│   └── lidar
│       ├── test
│       ├── train
│       └── val
├── car
│   ├── 3d_challenge
│   │   ├── train
│   │   └── val
│   └── lidar
│       ├── test
│       ├── train
│       └── val
├── CHANGELOG
├── goose_label_mapping.csv
├── LICENSE
├── spot
│   ├── 3d_challenge
│   │   ├── train
│   │   └── val
│   └── lidar
│       ├── test
│       ├── train
│       └── val
└── val_l
```

## Training Configurations

The repository includes configuration files for the solution:

*   **PPT with Decoupled Alignment**:
    `configs/goose/semseg-pt-v3m1-ppt_decoupled.py`


*   **PPT with Language-Driven Categorical Alignment**:
    `configs/goose/semseg-pt-v3m1-ppt.py`


## Pretrained Models

You can download the pretrained model weights from the link below:

[**Download Pretrained Models**](https://drive.google.com/drive/folders/15u5231sp7ROIie6SI_ELFk1wA_lO5owy?usp=sharing)

*(Note: This link is currently a placeholder and will be updated with the Google Drive link)*
