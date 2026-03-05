# CUDA MLP from Scratch: MNIST Classifier

A high-performance **Multi-Layer Perceptron (MLP)** implementation written in **C and CUDA**. This project moves beyond CPU-bound limits by offloading heavy matrix operations and backpropagation to the GPU, specifically designed to solve the MNIST handwritten digit classification problem.

## 🚀 Features
* **Massively Parallel Execution**: Forward and backward passes implemented via custom CUDA kernels.
* **GPU Memory Management**: Manual orchestration of `cudaMalloc`, `cudaMemcpy`, and `cudaMallocManaged` for efficient data throughput.
* **ReLU & Softmax Architecture**: Optimized for deep learning with **ReLU** activations in hidden layers and **Softmax** for multi-class probability distribution.
* **He Initialization**: Weights are scaled using $\sqrt{2/n}$ to maintain stable variance across layers.

## 🧠 The MNIST Challenge
The network is configured to process the **MNIST dataset**, consisting of 60,000 training images of handwritten digits (0-9). 

* **Input**: 784 neurons (28x28 pixel grayscale images).
* **Hidden Layers**: Fully configurable (Default: 128 → 64).
* **Output**: 10 neurons (One-hot encoded digits 0-9).



## 🛠️ Technical Implementation

### 1. Parallel Forward Pass
Each neuron's activation is calculated in parallel across GPU threads. The pre-activation sum is stored to facilitate the backward pass:
$$z_j = \text{bias}_j + \sum_{i=0}^{n} (\text{input}_i \cdot \text{weight}_{ij})$$

### 2. Loss & Optimization
* **Activation**: ReLU for hidden layers; Softmax for the output layer.
* **Loss Function**: Categorical Cross-Entropy for multi-class classification.
* **Optimizer**: Stochastic Gradient Descent (SGD) with parallel weight/bias updates via `update_kernel`.

### 3. Backpropagation
The error is propagated backward using custom kernels. For the output layer using Softmax and Cross-Entropy, the gradient simplifies to:
$$\delta_{output} = \text{Probs} - \text{Target}$$

For hidden layers, the error is calculated by accumulating gradients from the subsequent layer:
$$\delta_{hidden} = (\sum w_{next} \cdot \delta_{next}) \cdot \text{ReLU}'(z)$$



## 💻 Getting Started

### Prerequisites
* NVIDIA GPU with CUDA support.
* CUDA Toolkit (NVCC compiler).
* `mnist.c` and MNIST dataset files in the project directory.

### Compilation
Compile using the NVIDIA CUDA Compiler (`nvcc`):
```bash
nvcc main.cu -o mlp_cuda
