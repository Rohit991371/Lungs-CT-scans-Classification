# Lungs-CT-scans-Classification

# 🦠 COVID-19 Detection from Lung CT Scans using Attention-enhanced Xception Network

This project presents an advanced deep learning model for binary classification of lung CT scans into **COVID-positive** and **non-COVID** categories. Building upon the baseline proposed by **Chatterjee & Ghosh (2023)**, which used a deep CNN with 16-head channel attention, our model integrates **Xception** as the feature extractor with substantial architectural and training enhancements. These changes improve accuracy, reduce overfitting, and make the model highly robust for real-world deployment.

---

## 🚀 What's New Compared to the Baseline

| Feature | Chatterjee & Ghosh (2023) | **Proposed Model** |
|--------|-----------------------------|----------------------|
| Base Architecture | Generic Deep CNN | Xception (with depthwise separable convolutions) |
| Attention Mechanism | Multi-Head Channel Attention (16 heads) | Reduced to 8 heads (less redundancy, same performance) |
| Regularization | Limited | Batch Normalization + stronger regularization |
| Data Augmentation | Minimal | Extensive: rotation, zoom, shear, flips, normalization |
| Learning Rate Schedule | Static | Exponential Decay + ReduceLROnPlateau |
| Loss Function | Binary Crossentropy | Binary Crossentropy + Label Smoothing (0.1) |
| Accuracy | 96.99% | **97.33%** |

---

## 🧠 Model Architecture

- **Feature Extractor**: Xception
- **Attention Layer**: Custom 8-head Multi-Channel Attention
- **Regularization**: Dropout, Batch Normalization
- **Loss Function**: Binary Crossentropy with Label Smoothing
- **Optimizer**: Adam with Exponential LR Decay + ReduceLROnPlateau

---


