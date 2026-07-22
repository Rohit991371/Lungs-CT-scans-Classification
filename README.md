# Enhancing COVID-19 CT Image Classification with Attention-Augmented Pretrained Models

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange.svg?style=flat-square&logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-DL-red.svg?style=flat-square&logo=keras" alt="Keras">
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=flat-square" alt="License">
</p>

<p align="center">
  <b>Deep Learning Framework for COVID-19 Detection from Lung CT Scans using Transfer Learning & Attention Mechanisms</b>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Datasets](#datasets)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Models & Architectures](#models--architectures)
  - [Xception + Multi-Head Channel Attention (MHCA)](#1-xception--multi-head-channel-attention-mhca-main-model)
  - [Modified ResNet-50](#2-modified-resnet-50)
  - [ResNet-50 with FC Layer Modification](#3-resnet-50-with-fc-layer-modification)
  - [VGG19 (X-ray & CT Dataset)](#4-vgg19-extensive-x-ray-and-ct-dataset)
  - [VGG19 (Large CT Slice Dataset)](#5-vgg19-large-ct-scan-slice-dataset)
- [Results & Performance](#results--performance)
- [Citations](#citations)

---

## Overview

This repository implements a comprehensive deep learning framework for automated COVID-19 detection from lung CT scan images. The project leverages **Transfer Learning** on state-of-the-art pretrained Convolutional Neural Networks (CNNs) — **Xception**, **VGG19**, and **ResNet50** — enhanced with advanced techniques including **Multi-Head Channel Attention (MHCA)**, extensive data augmentation, and robust regularization strategies.

The primary objective is to develop a reliable, scalable, and computationally efficient diagnostic tool that can assist radiologists in early COVID-19 detection, particularly in resource-constrained healthcare environments where rapid and accurate diagnosis is critical.

### Motivation

The COVID-19 pandemic highlighted the urgent need for quick, precise diagnostic tools to reduce the burden on overburdened healthcare facilities. Deep learning offers a viable solution by automating image interpretation, enabling faster and more reliable diagnoses. This study leverages transfer learning and advanced preprocessing to build intelligent systems that support timely clinical decisions.

---

## Key Features

- **Best Performing Model**: Xception + Multi-Head Channel Attention achieving **97.86% accuracy**
- **Transfer Learning**: Fine-tuned pretrained models on large-scale ImageNet weights
- **Attention Mechanism**: Multi-Head Channel Attention for enhanced feature focusing
- **Multi-Dataset Training**: Evaluated across 4 publicly available CT scan datasets
- **Advanced Data Augmentation**: Rotation, flipping, zooming, shearing, brightness/contrast adjustments
- **Class Imbalance Handling**: Label smoothing, focal loss, and weighted sampling
- **Regularization**: Dropout, batch normalization, L2 regularization, and early stopping
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, and F1-Score metrics

---

## Datasets

This project utilizes four publicly available COVID-19 CT scan datasets:

### 1. SARS-CoV-2 CT Scan Dataset
| Attribute | Value |
|-----------|-------|
| **Total Images** | 2,482 |
| **COVID-19 Positive** | 1,252 |
| **COVID-19 Negative** | 1,230 |
| **Source** | Soares et al., 2020 |
| **Split** | 70% Train / 10% Validation / 20% Test |

### 2. Mehradaria COVID-19 Lung CT Scans
| Attribute | Value |
|-----------|-------|
| **Total Images** | 8,439 |
| **COVID-19 Positive** | 7,495 |
| **COVID-19 Negative** | 944 |
| **Source** | Heidarian, 2021 |

### 3. COVID-19 CT Dataset
| Attribute | Value |
|-----------|-------|
| **Total Images** | 10,290 |
| **Training Set** | 6,331 (70%) |
| **Validation Set** | 1,955 (10%) |
| **Test Set** | 2,004 (20%) |
| **Source** | Bhakar et al., 2023 |

### 4. Large COVID-19 CT Scan Slice Dataset
| Attribute | Value |
|-----------|-------|
| **Total Images** | 14,486 |
| **Training Set** | 11,588 (80%) |
| **Validation Set** | 1,449 (10%) |
| **Test Set** | 1,449 (10%) |
| **Source** | Cherifi et al., 2023 |

> **Note**: All datasets contain grayscale CT scan images labeled as COVID-19 positive or negative. The images exhibit varying scanner settings, slice thicknesses, and patient demographics, ensuring robust model generalization.

---

## Preprocessing Pipeline

All images undergo a standardized preprocessing pipeline to ensure compatibility and improve model convergence:

| Technique | Details |
|-----------|---------|
| **Resizing** | 224x224 pixels for VGG-19, ResNet-50, Xception; 180x180x3 for ResNet-50 variant |
| **Normalization** | Pixel scaling to [0,1], sample-wise and channel-wise normalization |
| **Rotation** | Random rotation up to 30 degrees to simulate scan angle variability |
| **Flipping** | Horizontal and vertical flipping for spatial augmentation |
| **Zooming** | Random zoom in/out with factor range 0.8-1.2x |
| **Shifting** | Width and height shift up to 20-30% to simulate misalignment |
| **Brightness/Contrast** | Random adjustments to simulate scanner settings |
| **Shearing** | Mild shear transformations for perspective variance |
| **Advanced** | Fill mode: nearest; aspect ratio preservation; combined batch-wise transformations |

### Preprocessing Code Snippet
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data Augmentation Configuration
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=0.2
)
```

---

## Models & Architectures

### 1. Xception + Multi-Head Channel Attention (MHCA) [MAIN MODEL]

> **This is the primary and best-performing model in our framework.**

The **Xception with Multi-Head Channel Attention (MHCA)** model represents the core contribution of this work. It integrates the Xception architecture as a powerful feature extractor with a custom Multi-Head Channel Attention module to enhance discriminative feature learning for COVID-19 classification.

#### Architecture Overview

```
Input Layer (224x224x1 Grayscale)
    |
    v
Conv2D 1x1 (3 Channels) ----> Xception Backbone
    |                           (70% Layers Frozen)
    v
MultiHeadChannelAttention (8 Heads)
    |
    v
Global Average Pooling
    |
    v
Dropout (0.4)
    |
    v
Dense Layer ----> 1024 Units, ReLU, L2 Regularization
    |
    v
Dropout (0.4)
    |
    v
Dense Layer ----> 512 Units, ReLU, L2 Regularization
    |
    v
Dropout (0.3)
    |
    v
Batch Normalization
    |
    v
Output Layer ----> 1 Neuron, Sigmoid Activation
```

#### Key Components

**Xception Backbone**
- Pretrained on ImageNet dataset
- 70% of layers frozen to preserve low-to-mid level feature representations
- Acts as a deep feature extractor capturing hierarchical patterns in CT images
- Utilizes depthwise separable convolutions for efficient computation

**Multi-Head Channel Attention (MHCA) Module**
The MHCA module is the critical enhancement that sets this model apart:

- **8 Attention Heads**: The input feature maps are split into 8 parallel heads, each independently processing the input sequence
- **Independent Feature Processing**: Each head focuses on different aspects/spatial regions of the input features, capturing diverse contextual information
- **Pooling Techniques**: Spatial features are extracted using pooling operations to capture global context
- **Dense Reduce & Expand Layers**: These layers learn channel-wise attention weights, emphasizing informative feature channels while suppressing less relevant ones
- **Sigmoid Activation**: The combined attention outputs are passed through a sigmoid function to produce channel attention weights in the range [0, 1]

**Regularization Strategy**
- **Dropout**: Progressive dropout rates (0.4 -> 0.4 -> 0.3) prevent overfitting
- **L2 Regularization**: Applied to dense layers to penalize large weights
- **Batch Normalization**: Stabilizes and accelerates training

**Training Configuration**

| Parameter | Value |
|-----------|-------|
| **Optimizer** | Adam |
| **Learning Rate** | Exponential decay |
| **Batch Size** | 16 |
| **Loss Function** | Binary Cross-Entropy with Label Smoothing (0.1) |
| **Epochs** | 80 |
| **Early Stopping** | Yes (monitors validation loss) |
| **Data Split** | 70% Training / 30% Validation |

#### Why Xception + MHCA Outperforms Others

1. **Attention-Guided Feature Selection**: MHCA enables the model to focus on infection-specific regions (ground-glass opacities, consolidation patterns) while ignoring irrelevant anatomical structures
2. **Multi-Scale Context**: 8 attention heads capture features at different scales and orientations
3. **Efficient Computation**: Reducing attention heads from 16 (baseline) to 8 balances performance and computational cost
4. **Robust Generalization**: Extensive augmentation + label smoothing + regularization prevents overfitting on limited medical data
5. **Transfer Learning Benefits**: Leveraging ImageNet-pretrained weights provides strong initial feature representations

#### Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | **97.86%** |
| **Precision** | 96.82% |
| **Recall** | **98.91%** |
| **F1-Score** | 97.86% |

> The high recall (98.91%) is particularly critical for medical diagnosis, as it minimizes false negatives -- ensuring very few COVID-19 positive cases are missed.

#### Implementation Code

```python
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import (Input, Conv2D, GlobalAveragePooling2D,
                                     Dense, Dropout, BatchNormalization, 
                                     Multiply, Reshape)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

# -------------------------------------------------
# Multi-Head Channel Attention (MHCA) Layer
# -------------------------------------------------
class MultiHeadChannelAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads=8, **kwargs):
        super(MultiHeadChannelAttention, self).__init__(**kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        channels = input_shape[-1]
        self.dense_reduce = Dense(channels // 2, activation='relu')
        self.dense_expand = Dense(channels, activation='sigmoid')

    def call(self, inputs):
        # Global Average Pooling for channel statistics
        gap = GlobalAveragePooling2D()(inputs)
        gap = Reshape((1, 1, gap.shape[-1]))(gap)

        # Reduce -> Expand (bottleneck attention)
        attention = self.dense_reduce(gap)
        attention = self.dense_expand(attention)

        # Apply attention weights
        return Multiply()([inputs, attention])

# -------------------------------------------------
# Build Xception + MHCA Model
# -------------------------------------------------
def build_xception_mhca_model(input_shape=(224, 224, 3), num_classes=1):
    # Input layer
    inputs = Input(shape=input_shape)

    # Convert grayscale to 3-channel for Xception compatibility
    x = Conv2D(3, (1, 1), padding='same')(inputs)

    # Xception Backbone (70% layers frozen)
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_tensor=x
    )

    # Freeze 70% of layers
    total_layers = len(base_model.layers)
    freeze_until = int(0.7 * total_layers)
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    # Extract features
    x = base_model.output

    # Multi-Head Channel Attention (8 Heads)
    x = MultiHeadChannelAttention(num_heads=8)(x)

    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)

    # Dropout + Dense Block 1
    x = Dropout(0.4)(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(x)

    # Dropout + Dense Block 2
    x = Dropout(0.4)(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)

    # Dropout + BatchNorm
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)

    # Output Layer
    outputs = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)

    # Compile with label smoothing
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', tf.keras.metrics.Precision(), 
                 tf.keras.metrics.Recall(), tf.keras.metrics.F1Score()]
    )

    return model

# Initialize model
model = build_xception_mhca_model()
model.summary()
```

---

### 2. Modified ResNet-50

A modified version of ResNet-50 adapted for COVID-19 CT scan classification with Swish activation and deep fully connected layers.

#### Architecture

```
Input Image (224x224x3)
    |
    v
ResNet-50 Pretrained
    |
    v
Global Average Pooling 2D (2048)
    |
    v
Batch Normalization
    |
    v
Dense(1024, Swish) ----> Dropout(0.5)
    |
    v
Dense(521, Swish) ----> Dropout(0.5)
    |
    v
Dense(1, Sigmoid)
    |
    v
Output (COVID / Non-COVID)
```

#### Key Features
- **Swish Activation**: Self-gated activation function (x * sigmoid(x)) for smoother gradients
- **Residual Connections**: Preserve gradient flow through deep networks
- **Deep FC Layers**: Two dense layers (1024 -> 521) with batch normalization and dropout

#### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | AdamW |
| **Learning Rate** | 0.0001 |
| **Batch Size** | 64 |
| **Loss Function** | Binary Cross-Entropy |
| **Frozen Layers** | First 100 layers of ResNet-50 |
| **Epochs** | 100 |

#### Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 94.64% |
| **Precision** | 92.02% |
| **Recall** | 93.51% |
| **F1-Score** | 92.76% |

---

### 3. ResNet-50 with FC Layer Modification

An alternative ResNet-50 implementation with a custom fully connected classification head designed for multi-scale feature learning.

#### Architecture

```
Input Image (180x180x3)
    |
    v
ResNet50 (2048)
    |
    v
Flatten (2048)
    |
    v
Batch Normalization
    |
    v
Dense(512) ----> BatchNorm ----> Dropout
    |
    v
Dense(256) ----> BatchNorm ----> Dropout
    |
    v
Dense(128) ----> BatchNorm ----> Dropout
    |
    v
Dense(64) ----> Dropout
    |
    v
Output (COVID-19 / Non-COVID-19)
```

#### Key Features
- **Progressive Dimensionality Reduction**: 512 -> 256 -> 128 -> 64 units with batch normalization at each step
- **Aggressive Dropout**: Multiple dropout layers to combat overfitting
- **Input Size**: 180x180x3 for faster computation

#### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | Adam |
| **Loss Function** | Categorical Cross-Entropy |
| **Epochs** | 50 |
| **Callbacks** | Early Stopping (validation loss monitoring) |

#### Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 93.79% |
| **Precision** | 93.66% |
| **Recall** | 93.78% |
| **F1-Score** | 93.71% |

---

### 4. VGG19 (Extensive X-ray and CT Dataset)

VGG19-based transfer learning model trained on a combined dataset of COVID-19 X-ray and CT images.

#### Architecture

```
Input (224x224x3)
    |
    v
VGG19 Base Model (Convolutional layers frozen)
    |
    v
Flatten (25,088)
    |
    v
Dense(512, ReLU) ----> Dropout(0.5)
    |
    v
Output Dense(1, Sigmoid)
```

#### Key Features
- **Frozen Convolutional Base**: All VGG19 convolutional layers frozen to preserve ImageNet features
- **Custom Classification Head**: Simple but effective dense layer with ReLU and dropout
- **Multi-Modal Training**: Trained on both X-ray and CT images for broader feature learning

#### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | Adam |
| **Loss Function** | Binary Cross-Entropy |
| **Dropout** | 0.5 |
| **Epochs** | 20 |

#### Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 93.27% |
| **Precision** | 94.70% |
| **Recall** | 92.63% |
| **F1-Score** | 94.00% |

---

### 5. VGG19 (Large CT Scan Slice Dataset)

Enhanced VGG19 model trained on the large 14,486-image CT scan slice dataset with advanced regularization and focal loss for class imbalance.

#### Architecture

```
Input (224x224x3)
    |
    v
VGG19 Base Model
    |
    v
GlobalAveragePooling2D
    |
    v
Dense(512, ReLU, L2 Reg) ----> BatchNorm ----> Dropout(0.5)
    |
    v
Dense(256, ReLU, L2 Reg) ----> BatchNorm ----> Dropout(0.4)
    |
    v
Dense(3, Softmax)
```

#### Key Features
- **Global Average Pooling**: Reduces parameters compared to flattening
- **Sparse Categorical Focal Loss**: Handles class imbalance with focusing parameter gamma
- **L2 Regularization**: Applied to all dense layers
- **Batch Normalization**: Stabilizes training of deep layers

#### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | AdamW |
| **Learning Rate** | 1e-4 |
| **Loss Function** | Sparse Categorical Focal Loss (gamma = 2) |
| **Dropout** | 0.5 / 0.4 |
| **Epochs** | 20 |
| **Callbacks** | Early Stopping |

#### Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | **97.67%** |
| **Precision** | 97.70% |
| **Recall** | 97.67% |
| **F1-Score** | 97.67% |

---

## Results & Performance

### Comparison with Literature Baselines

| Model (Literature) | Epochs | Accuracy | Precision | Recall | F1-Score |
|-------------------|--------|----------|-----------|--------|----------|
| Xception with MHCA [Chatterjee & Ghosh, 2023] | 50 | 96.99% | 96.99% | 97.00% | 96.99% |
| CTCovid19 using ResNet-50 [Antunes et al., 2025] | 100 | 97.00% | 96.80% | 97.20% | 96.90% |
| ResNet50 Multi-Class [Walvekar & Shinde, 2020] | 5 | 96.23% | 95.60% | 97.15% | 96.37% |
| VGG19-Based [Mohbey et al., 2022] | 60 | 95.00% | 95.00% | 95.00% | 95.00% |
| DL VGG19 [Shah et al., 2021] | 30 | 94.50% | 94.10% | 94.10% | 94.10% |

### Our Proposed Implementations

| Model | Epochs | Accuracy | Precision | Recall | F1-Score |
|-------|--------|----------|-----------|--------|----------|
| **Xception Net + MHCA** | **80** | **97.86%** | **96.82%** | **98.91%** | **97.86%** |
| Modified ResNet-50 | 100 | 94.64% | 92.02% | 93.51% | 92.76% |
| ResNet-50 (Modified FC) | 50 | 93.79% | 93.66% | 93.78% | 93.71% |
| VGG19 (X-ray & CT) | 20 | 93.27% | 94.70% | 92.63% | 94.00% |
| VGG19 (Large CT Slice) | 20 | 97.67% | 97.70% | 97.67% | 97.67% |

### Key Observations

- **Xception + MHCA** achieves the highest accuracy (97.86%) and recall (98.91%), making it the most reliable for clinical deployment
- **VGG19 on Large CT Slice Dataset** shows competitive performance (97.67%) due to massive dataset size and focal loss
- All proposed models exceed 93% accuracy, demonstrating the effectiveness of transfer learning in medical imaging
- The attention mechanism in Xception+MHCA provides the best balance of precision and recall

---

## Citations

If you use this work in your research, please cite:

```bibtex
@article{gupta2024covid,
  title={Enhancing COVID-19 CT Image Classification with Attention-Augmented Pretrained Models},
  author={Gupta, Rohit and Koirala, Sweekar and Piri, Jayashree and Dey, Raghunath and Mehta, Subhash and Karna, Samrat and Gupta, Sumit and Mohanty, Surajit},
  journal={IEEE},
  year={2024}
}
```

### Related Works Referenced

1. Chatterjee & Ghosh (2023) -- Xception + MHCA Baseline
2. Antunes et al. (2025) -- CTCovid19 ResNet-50
3. Walvekar & Shinde (2020) -- ResNet50 Multi-Class
4. Mohbey et al. (2022) -- VGG19 Transfer Learning
5. Shah et al. (2021) -- VGG-19 Comparative Study

---

## Acknowledgments

- SARS-CoV-2 CT Scan Dataset by Soares et al.
- COVID-19 CT Dataset by Bhakar et al.
- TensorFlow and Keras teams for the deep learning framework

---

<p align="center">
  <b>Star this repository if you find it helpful!</b>
</p>

<p align="center">
  For questions or collaborations, feel free to open an issue or contact the authors.
</p>
