#ifndef KERNELS_CUH_
#define KERNELS_CUH_

// Include cuda_runtime to ensure __global__ and other types are recognized
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define EPSILON 1e-15

#define CUDA_CHECK_ERR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

typedef struct {
    // Dimentions
    int input_size;
    int output_size;

    // Parameters
    float *weights;
    float *biases;

    float *preactivations;
    float *activations;

    // Gradients in the layer
    float *grad_wrt_w;
    float *grad_wrt_b;
    float *grad_wrt_input;

    // Momentum parameters for adam optimizer
    float *m_w;
    float *v_w;
    float *m_b;
    float *v_b;
} Layer;
/**
 */
__global__ void safe_softmax_kernel(float* input, int size, float* output, int batch_size);

/* Softmax kernel for SGD */
__global__ void softmax_kernel(float* input, int size, float* output_dest);

__global__ void forward_kernel2d(float *input, float *weights, float *biases,
                                 float *preactivations, float *activations,
                                 int input_size, int output_size,
                                 int batch_size);

__global__ void forward_kernel(float *d1, float *d2, float *d3, float *d4,
                               float *d5, int i1, int i2);

__global__ void zero_layer_gradients(float *grad_wtr_b, float *grad_wtr_input,
                                     int size);

__global__ void zero_weight_gradients(float *weights, int size);

__global__ void backward_kernel_output_layer2d(
    float *grad_wtr_b, float *grad_wtr_w, float *prev_grad_wrt_input,
    float *previous_activation, float *weight, int input_size,
    int output_size, int batch_size, int *target, float *probs);

__global__ void backward_kernel_output_layer(
    float *grad_wrt_b, float *grad_wrt_w, float *previous_grad_wrt_input,
    float *previous_activations, float *weights, int input_size,
    int output_size, int target, float *probs);

__global__ void backward_kernel2d(Layer *layer, Layer *previous_layer,
                                  float *initial_input, int current_layer_idx,
                                  int batch_size);

__global__ void sparce_categorical_cross_entropy_kernel(float* probabilities, int* target_labels,
                                                 int batch_size, int num_classes, float* loss);

__global__ void backward_kernel(Layer *layer, Layer *previous_layer,
                                float *initial_inputs, int current_layer_idx);


__global__ void update_biases_1d_kernel(float *biases, float *grad_b,
                                        int output_size, float learning_rate,
                                        int batch_size);

__global__ void update_weights_2d_kernel(float* weights, float* grad_w,
                                         int input_size, int output_size,
                                         float learning_rate, int batch_size);

__global__ void update_kernel_minibatch(float* weights, float* biases, float* grad_w,
                                        float* grad_b, int input_size, int out_size,
                                        float learning_rate, int batch_size);

__global__ void update_kernel(float *weights, float *biases, float *grad_w,
                              float *grad_b, int in_size, int out_size,
                              float learning_rate);

#endif // KERNELS_CUH_
