#include "kernels.cuh"


/* Softmax with minibatch */
__global__ void safe_softmax_kernel(float* input, int size, float* output_probs, int batch_size) {

    // Take the index of the image in the batch
    int batch_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (batch_idx < batch_size) {
        // take the stating place of the 1D array
        int starting_idx = batch_idx * size;

        // Initialize the max value as the first index
        float max_val = input[starting_idx];
        // Find the max value
        for(int i = 1; i < size; i++) {
            if(input[starting_idx + i] > max_val) {
                max_val = input[starting_idx + i];
            }
        }

        // Calculate the sum of exponents in softmax
        float sum = 0.0;
        for (int i = 0; i < size; i++) {
            output_probs[starting_idx + i] = exp(input[starting_idx + i] - max_val); // Subtract max for stability
            sum += output_probs[starting_idx + i];
        }

        // Calculate the individial probabilities for output
        for (int i = 0; i < size; i++){
            output_probs[starting_idx + i] /= sum;
        } 
        // for checking values for debugging
        // printf("%.4f\n", output_probs[starting_idx]);
    }
}

// Implementation of softmax layer
__global__ void softmax_kernel(float* input, int size, float* output_dest) {

    float max_val = input[0];
    for(int i = 1; i < size; i++) if(input[i] > max_val) max_val = input[i];

    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        output_dest[i] = exp(input[i] - max_val); // Subtract max for stability
        sum += output_dest[i];
    }
    for (int i = 0; i < size; i++) output_dest[i] /= sum;
}


/**
 * 2D forward pass for minibatches
 */
__global__ void forward_kernel2d(float *input, float *weights, float *biases,
                            float *preactivation, float *activation, int input_size,
                            int output_size, int batch_size)
{
    // Map threads to asoecific Neuron x and a specific image y
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockDim.y * blockIdx.y + threadIdx.y;
    
    // Check bounds
    if ( neuron_idx < output_size && batch_idx < batch_size ) {

        float sum = biases[neuron_idx];

        for ( int i = 0; i < input_size; i++ ) {
            int input_idx = batch_idx * input_size + i;

            int weight_idx = i * output_size + neuron_idx;

            sum += input[input_idx] * weights[weight_idx];
        }

        int out_ptr = (batch_idx * output_size) + neuron_idx;
        preactivation[out_ptr] = sum;

        // debugging
        // printf("Neuron number %d in batch number %d preactivation %f\n", neuron_idx+1, batch_idx+1,
               // preactivation[out_ptr]);

        // skip relu at the output layer
        if (output_size == 10) {
            activation[out_ptr] = preactivation[out_ptr];
        }
        else {
            activation[out_ptr] = sum > 0 ? sum : 0; // ReLU activation
        }
    }
}


/* Forward pass kernel for a layer */
__global__ void forward_kernel(float *input, float *weights, float *biases,
                            float *preactivation, float *activation,
                            int input_size, int output_size) {

    // determing threads to calculate each neuron
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Only work if this thread is corresponding to a valid neuron
    if (neuron_idx < output_size) {
        float sum = biases[neuron_idx];

        for (int i = 0; i < input_size; i++) {
            sum += input[i] * weights[i * output_size + neuron_idx];
        }
        preactivation[neuron_idx] = sum; // store the preactivation for backward pass

        // skip relu at the output layer
        if (output_size == 10) {
            activation[neuron_idx] = preactivation[neuron_idx];
        }
        else {
            activation[neuron_idx] = sum > 0 ? sum : 0; // ReLU activation
        }
    }
}

/* Zero gradients  */
__global__ void zero_layer_gradients(float *grad_b, float *grad_in, int num_nodes) {
    int output_dim_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (output_dim_idx < num_nodes) {
        grad_b[output_dim_idx] = 0;
        grad_in[output_dim_idx] = 0;
    }
}

// Dedicated Kernel for Weight Gradients (Size: input_size * output_size)
__global__ void zero_weight_gradients(float *grad_w, int total_weights) {
    int weight_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (weight_idx < total_weights) {
        grad_w[weight_idx] = 0.0;
    }
}

__global__ void backward_kernel_output_layer2d(
    float *grad_wtr_b, float *grad_wtr_w, float *previous_grad_wtr_input,
    float *previous_activiation, float *weight, int input_size,
    int output_size, int batch_size, int *target, float *probs) {

  int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

  if (neuron_idx < output_size && batch_idx < batch_size) {

    int probs_idx = batch_idx * output_size + neuron_idx;
    // Calculate derivative for the softmax index (probs_idx)
    // if the neuron index is the same as the target then pass the error backwards
    float target_val = (neuron_idx == target[batch_idx]) ? 1.0 : 0.0;
    float delta = probs[probs_idx] - target_val;

    atomicAdd(&grad_wtr_b[neuron_idx], delta);

    for (int i = 0; i < input_size; i++) {
      int weight_idx = i * output_size + neuron_idx;
      int act_idx = batch_idx * input_size + i;

      atomicAdd(&grad_wtr_w[weight_idx], delta * previous_activiation[act_idx]);
      atomicAdd(&previous_grad_wtr_input[act_idx], delta * weight[weight_idx]);
    }
  }
}

__global__ void backward_kernel_output_layer(
    float *grad_wrt_b, float *grad_wrt_w, float *previous_grad_wrt_input,
    float *previous_activations, float *weights, int input_size,
    int output_size, int target, float *probs) {

    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (neuron_idx < output_size) {
        // Shortcut for the derivative of the cross entropy loss with respect to the
        // input of softmax function Every other index other that the one that
        // corresponds to the target index is 0
        float target_val = (neuron_idx == target) ? 1.0 : 0.0;
        float delta = probs[neuron_idx] - target_val;

        grad_wrt_b[neuron_idx] = delta;

        for (int k = 0; k < input_size; k++) {
            int weight_idx = k * output_size + neuron_idx;
            grad_wrt_w[weight_idx] = delta * previous_activations[k];
            // Use atomicAdd to prevent race conditions from multiple 'j' threads
            atomicAdd(&previous_grad_wrt_input[k], delta * weights[weight_idx]);
        }
    }
}

__global__ void backward_kernel2d(Layer *layer, Layer *previous_layer,
                                  float *initial_input, int current_layer_idx,
                                  int batch_size)
{
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (neuron_idx < layer->output_size && batch_idx < batch_size) {
        float relu_derivation = layer->activations[batch_idx * layer->output_size + neuron_idx] > 0 ? 1.0 : 0.0;
        float delta = layer->grad_wrt_input[batch_idx * layer->output_size + neuron_idx] * relu_derivation;

        atomicAdd(&layer->grad_wrt_b[neuron_idx], delta);

        for ( int i = 0; i < layer->input_size; i++ ) {
            int weight_idx = (i * layer->output_size) + neuron_idx;
            int prev_act_idx = (batch_idx * layer->input_size) + i;

            float prev_act = (current_layer_idx == 0) ? initial_input[prev_act_idx] : previous_layer->activations[prev_act_idx];

            atomicAdd(&layer->grad_wrt_w[weight_idx], delta * prev_act);
            
            // FIXED: Only pass error backward if we aren't at the input layer
            if (current_layer_idx != 0) {
                atomicAdd(&previous_layer->grad_wrt_input[prev_act_idx], delta * layer->weights[weight_idx]);
            }
        }
    }
}

__global__ void sparce_categorical_cross_entropy_kernel(float* probabilities, int* target_labels,
                                                 int batch_size, int num_classes, float *loss) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size) {
        int target = target_labels[batch_idx];

        int target_idx_probs = (batch_idx * num_classes) + target;

        float p = probabilities[target_idx_probs];

        if (p < EPSILON) p = EPSILON;
        
        atomicAdd(loss, -log(p));
    }
}

__global__ void backward_kernel(Layer* layer, Layer* previous_layer, float* initial_inputs, int current_layer_idx) {
    
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (neuron_idx < layer->output_size) {
        float delta = 0;

        float relu_derivation = layer->activations[neuron_idx] == 0 ? 0.0 : 1.0; // If relu is not 0 then gradient is 1
        delta = layer->grad_wrt_input[neuron_idx] * relu_derivation;

        layer->grad_wrt_b[neuron_idx] = delta;

        for ( int input_size_idx = 0; input_size_idx < layer->input_size; input_size_idx++ ) {
            // weights are [input_size * output_size]
            // Accessing weights for input j and neuron j:
            int weight_idx = input_size_idx * layer->output_size + neuron_idx;

            // Gradient is delta * activation of the previous layer
            float *prev_act = (current_layer_idx == 0) ? initial_inputs : previous_layer->activations;
            layer->grad_wrt_w[weight_idx] = delta * prev_act[input_size_idx];

            // Pass error back to the previous layer
            if (current_layer_idx != 0) {
                previous_layer->grad_wrt_input[input_size_idx] += delta * layer->weights[weight_idx];
            }
        }
    }
}

__global__ void update_biases_1d_kernel(float *biases, float *grad_b,
                                        int output_size, float learning_rate,
                                        int batch_size)
{
    int neuron_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (neuron_idx < output_size) {
        grad_b[neuron_idx] -= learning_rate * (grad_b[neuron_idx] / batch_size);
    }
}

__global__ void update_weights_2d_kernel(float* weights, float* grad_w,
                                         int input_size, int output_size,
                                         float learning_rate, int batch_size)
{
    int neuron_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int input_idx = blockDim.y * blockIdx.y + threadIdx.y;

    if (neuron_idx < output_size && input_idx < input_size) {
        int weight_idx = input_idx * output_size + neuron_idx;

        weights[weight_idx] -= learning_rate * (grad_w[weight_idx] / batch_size);
    }
}

__global__ void update_kernel_minibatch(float* weights, float* biases, float* grad_w,
                                        float* grad_b, int input_size, int out_size,
                                        float learning_rate, int batch_size)
{
    int neuron_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (neuron_idx < out_size) {
        // Update parematers with gradients and nromalize them with the batch size
        biases[neuron_idx] -= learning_rate * (grad_b[neuron_idx] / batch_size);

        for (int input_size_idx = 0; input_size_idx < input_size; input_size_idx++) {

            // Calculate the index of the weight input_idx * output_size + neuron_index
            int weight_idx = input_size_idx * out_size + neuron_idx;
            // Update weights and normalize with the batch size
            weights[weight_idx] -= learning_rate * (grad_w[weight_idx] / batch_size);
        }
    }
}

__global__ void update_kernel(float* weights, float* biases, float* grad_w, float* grad_b, 
                             int in_size, int out_size, float learning_rate) {
    int neuron_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (neuron_idx < out_size) {
        biases[neuron_idx] -= learning_rate * grad_b[neuron_idx];

        for (int input_size_idx = 0; input_size_idx < in_size; input_size_idx++) {

            // Calculate the index of the weight input_idx * output_size + neuron_index
            int weight_idx = input_size_idx * out_size + neuron_idx;
            weights[weight_idx] -= learning_rate * grad_w[weight_idx];
        }
    }
}
