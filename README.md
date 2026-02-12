# Minimalist MLP from Scratch in C

A lightweight, dependency-free implementation of a **Multi-Layer Perceptron (MLP)** written entirely in C. This project demonstrates the core mechanics of deep learning, including manual memory management, forward propagation, backpropagation using the chain rule, and gradient descent.



## 🚀 Features
* **Pure C Implementation**: No external machine learning libraries.
* **Dynamic Layering**: Support for configurable input, hidden, and output layer sizes via struct-based architecture.
* **Binary Cross-Entropy (BCE)**: Loss function optimized for binary classification.
* **Sigmoid Activation**: Custom implementation of the Sigmoid function and its derivative.
* **Manual Backpropagation**: A from-scratch implementation of the chain rule to update weights and biases.
* **Memory Management**: Explicit allocation of weights, biases, and gradients on the heap.

## 🧠 The XOR Challenge
The network is currently configured to solve the **XOR (Exclusive OR) problem**. XOR is a classic benchmark in machine learning because it is not linearly separable, meaning a single-layer perceptron cannot solve it. This implementation uses a hidden layer to learn the non-linear decision boundary.

| Input A | Input B | Output |
| :--- | :--- | :--- |
| 0 | 0 | **0** |
| 0 | 1 | **1** |
| 1 | 0 | **1** |
| 1 | 1 | **0** |

## 🛠️ Mathematical Implementation

### 1. Forward Pass
For each layer, the network calculates the pre-activation sum $z$ and then applies the activation function $\sigma$:

$$z = \sum (w_i \cdot x_i) + b$$
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$



### 2. Backpropagation
The error is propagated backward using the derivative of the loss with respect to the weights. For the output layer using BCE and Sigmoid, the gradient simplifies to:
$$\delta = \text{Output} - \text{Target}$$

For hidden layers, we accumulate the gradients from the layer above:
$$\delta_{hidden} = (\sum w_{next} \cdot \delta_{next}) \cdot \sigma'(z)$$

### 3. Optimization
Weights are updated using Stochastic Gradient Descent (SGD):
$$W_{new} = W_{old} - (\eta \cdot \text{Gradient})$$
*Note: This project found that a learning rate ($\eta$) of ~0.1 is optimal for XOR convergence to avoid overshooting.*



## 💻 Getting Started

### Compilation
Compile using GCC (ensure the math library is linked with `-lm`):
```bash
gcc main.c -Wall -Wextra -o mlp_program -lm
