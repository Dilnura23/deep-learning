# CS371: Deep Learning Assignments

This repository contains a collection of programming assignments for the **CS371 Introduction to Deep Learning** course. The projects range from implementing fundamental neural network components from scratch to building advanced architectures for Computer Vision and Natural Language Processing using PyTorch.

## 📂 Repository Structure

| File name | Topic |
| :--- | :---  |
| **`multi-layer-perceptron.ipynb`** | Multi-Layer Perceptron (From Scratch) |
| **`CNN_image_classification.ipynb`** | CNNs (ResNet, Inception) on CIFAR-10 |
| **`image_segmentation_FCN.ipynb`** | Semantic Segmentation (FCN) |
| **`Neural_Machine_Translation.ipynb`** | NMT using Seq2Seq & Attention |

---

## 📝 Assignment Details

### 1. Introduction to Deep Learning (MLP)
**Filename:** `multi-layer-perceptron.ipynb`

An implementation of a Multi-Layer Perceptron (MLP) using only **NumPy**. This assignment focuses on understanding the mathematical foundations of deep learning.

* **Key Tasks:**
    * Implementing Forward Propagation (Linear layers, ReLU).
    * Deriving and implementing **Backpropagation** (gradients for weights/biases).
    * Implementing the **Gradient Descent** optimizer.
    * Training on a synthetic binary classification dataset.

### 2. Image Classification using CNNs
**Filename:** `CNN_image_classification.ipynb`

A comprehensive exploration of Convolutional Neural Networks (CNNs) for image classification on the **CIFAR-10** dataset.

* **Key Tasks:**
    * Building modular blocks: **Convolutional**, **Residual (Plain & Bottleneck)**, and **Inception** blocks.
    * Constructing `MyResNet` and `MyInception` architectures using `torch.nn.Module`.
    * Comparing model performance (Accuracy vs. Parameter count).
    * **Result:** Achieved ~88% accuracy with ResNet variants.

### 3. Semantic Segmentation using FCN
**Filename:** `image_segmentation_FCN.ipynb`

Implementation of Fully Convolutional Networks (FCN) for semantic image segmentation on the **PASCAL VOC 2011** dataset (augmented with SBD).

* **Key Tasks:**
    * Fine-tuning a pretrained **VGG-16** backbone.
    * Implementing **Transposed Convolutions** for upsampling.
    * Building **FCN-32s** and **FCN-8s** architectures with skip connections.
    * Evaluating performance using the Intersection over Union (IoU) metric.

### 4. Neural Machine Translation (NMT)
**Filename:** `Neural_Machine_Translation.ipynb`

Development of a Sequence-to-Sequence (Seq2Seq) model for language translation (e.g., English to French).

* **Key Tasks:**
    * Data preprocessing: Tokenization and building vocabulary.
    * Implementing **Encoder-Decoder** architectures using **LSTMs**.
    * Integrating **Attention Mechanisms** to handle long-range dependencies.
    * (Optional) Exploring Transformer components.

---

## 🛠️ Tech Stack & Requirements

* **Language:** Python 3.x
* **Frameworks:** PyTorch, NumPy
* **Libraries:** `torchvision`, `matplotlib`, `scikit-learn`, `tqdm`
* **Platform:** Optimized for Google Colab (GPU Runtime recommended)

## 🚀 How to Run

1.  Clone this repository:
    ```bash
    git clone [https://github.com/yourusername/CS371-Deep-Learning.git](https://github.com/yourusername/CS371-Deep-Learning.git)
    ```
2.  Open the desired notebook (`.ipynb`) in **Google Colab** or a local Jupyter environment.
3.  **Note:** Some assignments (specifically #3) require downloading pretrained weights (e.g., VGG-16) or mounting Google Drive to access datasets. Check the "Instructions" cell at the top of each notebook for specific setup steps.

## 📜 License

This project is for educational purposes as part of the CS371 course at KAIST.
