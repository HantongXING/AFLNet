# AFLNet: Auxiliary Feature Learning-Guided Cross-Channel Automatic Modulation Classification

This repository provides the official PyTorch implementation of **AFLNet**, proposed in the paper:

AFLNet: Auxiliary Feature Learning-Guided Cross-Channel Automatic Modulation Classification  
Hantong Xing, Shuang Wang, Chenxu Wang, Dou Quan, et al.  
IEEE Transactions on Communications, 2025

This code targets cross-channel automatic modulation classification (AMC) under complex wireless channel variations, including AWGN, Rician, and Rayleigh channels.

------------------------------------------------------------

## 1. Introduction

In practical wireless communication scenarios, transmitted signals are affected by various channel impairments such as multipath fading, Doppler shift, carrier frequency offset, and sampling rate offset. These impairments lead to severe performance degradation when a model trained on one channel condition is directly applied to another.

AFLNet addresses this problem by explicitly enhancing the discriminability of the target-domain feature space before and during domain alignment, enabling robust cross-channel modulation classification.

------------------------------------------------------------

## 2. Method Overview

AFLNet consists of two main components.

### 2.1 Auxiliary Feature Learning (AFL)

Auxiliary feature learning is applied only to target-domain data and aims to improve the intrinsic structure of the target feature space.

Similarity-based feature learning:
- Contrastive learning between original and augmented target samples
- Encourages intra-class compactness

Confidence-based feature learning:
- Information Maximization (IM) loss
- Promotes confident predictions and global class diversity

### 2.2 Collaborative Feature Alignment

Two complementary alignment strategies are jointly employed.

Adversarial domain alignment:
- Domain-Adversarial Neural Network (DANN)
- Aligns global feature distributions between source and target domains

Self-training with EMA teacher:
- Exponential Moving Average (EMA) teacher network
- Generates pseudo-labels for target-domain samples
- Enables class-level feature alignment

During inference, only the student network is retained.

------------------------------------------------------------

## 3. Code Structure

Project directory structure:

    .
    ├── AFLNet_main.py          # Main training and evaluation script
    ├── CLDNN.py                # Network architectures (GNET, DNET, DANN)
    ├── teachermodel.py         # EMA Teacher for self-training
    ├── selfsup/
    │   └── ntx_ent_loss.py     # NT-Xent contrastive loss
    └── README.md               # This file

------------------------------------------------------------

## 4. Network Architecture

### 4.1 GNET (Student Network)

Defined in CLDNN.py.

GNET jointly serves as:
- Feature extractor
- Modulation classifier
- Projection head for contrastive learning

Main components include:
- 1D CNN encoder for IQ signal feature extraction
- Additional convolution layers
- Two-layer LSTM for temporal modeling
- Fully connected classifier
- Projection head for similarity-based learning

Forward outputs of GNET:

    features, logits, reconstruction, projection

------------------------------------------------------------

### 4.2 Domain Discriminator (DANN)

Defined as DNET_DANN in CLDNN.py.

- Uses a Gradient Reversal Layer (GRL)
- Performs adversarial training to align source and target features
- Operates on deep features extracted by GNET

------------------------------------------------------------

### 4.3 EMA Teacher

Defined in teachermodel.py.

The EMA teacher:
- Maintains an exponential moving average of the student network
- Generates pseudo-labels for target-domain samples
- Weights pseudo-label loss using prediction confidence

The EMA teacher implementation follows the paradigm introduced in:

MIC: Masked Image Consistency for Context-Enhanced Domain Adaptation  
Hoyer L., Dai D., Wang H., et al., CVPR 2023

------------------------------------------------------------

## 5. Loss Functions

The overall training objective corresponds to Eq. (12) in the paper and includes the following components.

Source-domain classification loss:
- Cross-entropy loss on labeled source-domain data

Similarity-based auxiliary loss:
- NT-Xent contrastive loss between original and augmented target samples

Confidence-based auxiliary loss:
- Information Maximization (IM) loss
- Entropy minimization and batch-level diversity maximization

Collaborative alignment loss:
- Adversarial domain loss (DANN)
- Pseudo-label supervised loss on augmented target samples

All loss terms are jointly optimized during training.

------------------------------------------------------------

## 6. Data Augmentation

All augmentations are applied to target-domain data only and include:
- Signal inversion
- Amplitude scaling
- Temporal shifting
- Phase rotation in the IQ domain
- Random temporal masking

These augmentations support auxiliary feature learning and self-training.

------------------------------------------------------------

## 7. Training and Evaluation

### 7.1 Training

The training procedure is implemented in AFLNet_main.py.

- Source domain: supervised learning
- Target domain: unsupervised auxiliary feature learning and alignment
- EMA teacher updated at each iteration
- Best model selected based on target-domain accuracy

### 7.2 Evaluation

During evaluation:
- Only the student network (GNET) is used
- Performance is reported in terms of overall accuracy and SNR-wise accuracy

------------------------------------------------------------

## 8. Requirements

- Python 3.7 or later
- PyTorch 1.7 or later
- NumPy
- timm

------------------------------------------------------------

## 9. Usage

1. Prepare datasets in the same format as the RadioML dataset.
2. Update dataset paths in AFLNet_main.py.
3. Run training:

    python AFLNet_main.py

4. The best model will be saved as:

    AFLNet.pth

------------------------------------------------------------

## 10. Citation

If you use this code in your research, please cite:

Xing H, Wang S, Wang C, et al.  
AFLNet: Auxiliary Feature Learning-Guided Cross-Channel Automatic Modulation Classification.  
IEEE Transactions on Communications, 2025.

BibTeX entry:

    @article{xing2025aflnet,
      author  = {Xing, Hantong and Wang, Shuang and Wang, Chenxu and others},
      title   = {AFLNet: Auxiliary Feature Learning-Guided Cross-Channel Automatic Modulation Classification},
      journal = {IEEE Transactions on Communications},
      year    = {2025}
    }

------------------------------------------------------------

## 11. Notes

- This repository focuses on faithful reproduction of the AFLNet framework described in the paper.
- No additional architectural modifications or optimizations are applied beyond those reported.
- The EMA teacher component is adapted from prior work and is not an original contribution of AFLNet.


## Acknowledgements

Some components in this repository are adapted from prior work:

- The self-supervised contrastive loss (NT-Xent) implementation in `selfsup/` is adapted from:
  
  Bai J, Wang X, Xiao Z, et al.  
  Achieving efficient feature representation for modulation signal: A cooperative contrast learning approach.  
  IEEE Internet of Things Journal, 2024, 11(9): 16196–16211.

- The EMA teacher–student framework used for pseudo-label generation is adapted from:
  
  Hoyer L, Dai D, Wang H, et al.  
  MIC: Masked Image Consistency for Context-Enhanced Domain Adaptation.  
  Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

These components are used as implementation building blocks.  
The overall AFLNet framework, auxiliary feature learning strategy, and collaborative alignment mechanism are proposed in our work.
