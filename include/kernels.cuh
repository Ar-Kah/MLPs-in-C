#ifndef KERNELS_CUH_
#define KERNELS_CUH_

// Include cuda_runtime to ensure __global__ and other types are recognized
#include <cuda_runtime.h>

// Function prototypes must end with a semicolon!
__global__ void softmax_kernel(double* input, int size, double* output, int stride);

__global__ void forward_kernel2d(double* d1, double* d2, double* d3, double* d4, double* d5, int i1, int i2, int i3);

__global__ void forward_kernel(double* d1, double* d2, double* d3, double* d4, double* d5, int i1, int i2);

__global__ void zero_layer_gradients(double* d1, double* d2, int size);

__global__ void zero_weight_gradients(double* d1, int size);

__global__ void backward_kernel_output_layer(
    double* grad_wrt_b, 
    double* grad_wrt_w, 
    double* previous_grad_wrt_input,
    double* previous_activations,
    double* weights,
    int input_size, 
    int output_size, 
    int target, 
    double *probs);

__global__ void backward_kernel2D(Layer *layer, Layer *previous_layer, double *initial_input, int current_layer_idx);

__global__ void backward_kernel(Layer* layer, Layer* previous_layer, double* initial_inputs, int current_layer_idx);

__global__ void update_kernel(double* weights, double* biases, double* grad_w, double* grad_b, 
                             int in_size, int out_size, double learning_rate);

#endif // KERNELS_CUH_
