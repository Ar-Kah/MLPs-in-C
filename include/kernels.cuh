#ifndef KERNELS_CUH_
#define KERNELS_CUH_

// Include cuda_runtime to ensure __global__ and other types are recognized
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

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
    int input_size;
    int output_size;
    double *weights;
    double *biases;
    double *preactivations;
    double *activations;
    double *grad_wrt_w;
    double *grad_wrt_b;
    double *grad_wrt_input;
} Layer;
/**
 */
__global__ void safe_softmax_kernel(double* input, int size, double* output, int batch_size);

/* Softmax kernel for SGD */
__global__ void softmax_kernel(double* input, int size, double* output_dest);

__global__ void forward_kernel2d(double *input, double *weights, double *biases,
                                 double *preactivations, double *activations,
                                 int input_size, int output_size,
                                 int batch_size);

__global__ void forward_kernel(double *d1, double *d2, double *d3, double *d4,
                               double *d5, int i1, int i2);

__global__ void zero_layer_gradients(double *grad_wtr_b, double *grad_wtr_input,
                                     int size);

__global__ void zero_weight_gradients(double *weights, int size);

__global__ void backward_kernel_output_layer2d(
    double *grad_wtr_b, double *grad_wtr_w, double *prev_grad_wrt_input,
    double *previous_activation, double *weight, int input_size,
    int output_size, int batch_size, int *target, double *probs);

__global__ void backward_kernel_output_layer(
    double *grad_wrt_b, double *grad_wrt_w, double *previous_grad_wrt_input,
    double *previous_activations, double *weights, int input_size,
    int output_size, int target, double *probs);

__global__ void backward_kernel2d(Layer *layer, Layer *previous_layer,
                                  double *initial_input, int current_layer_idx,
                                  int batch_size);

__global__ void sparce_categorical_cross_entropy_kernel(double* probabilities, int* target_labels,
                                                 int batch_size, int num_classes, double* loss);

__global__ void backward_kernel(Layer *layer, Layer *previous_layer,
                                double *initial_inputs, int current_layer_idx);

__global__ void update_kernel(double *weights, double *biases, double *grad_w,
                              double *grad_b, int in_size, int out_size,
                              double learning_rate);

#endif // KERNELS_CUH_
