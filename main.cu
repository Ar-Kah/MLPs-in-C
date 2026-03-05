#include <string.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

#include "mnist.c"

// define macro for getting the size of a list of integers
#define LEN(x) (sizeof(x) / sizeof(x[0]))

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

typedef struct {
    Layer *layers;
    int num_layers;
} Network;


/* After taking the highest index after applying softmax
 * all but one index is 0 so we only calculate the loss
 * for that index */
double sparce_categorical_cross_entropy(double *predictions, int label_index) {
    // prediction is the array from softmax
    // label_index is the actual number of 0-9
    double p = predictions[label_index];

    if (p < 1e-15) p = 1e-15;

    return -log(p);
}

// The derivative of BCE Loss with respect to the Sigmoid activation
double bce_derivative(double target, double prediction) {
    // We add a tiny epsilon to prevent division by zero
    double epsilon = 1e-7f;
    if (prediction < epsilon) prediction = epsilon;
    if (prediction > 1.0f - epsilon) prediction = 1.0f - epsilon;

    return (prediction - target) / (prediction * (1.0f - prediction));
}


// Implementation of softmax layer
__global__ void softmax_kernel(double* input, int size, double* output_dest) {

    double max_val = input[0];
    for(int i = 1; i < size; i++) if(input[i] > max_val) max_val = input[i];

    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        output_dest[i] = exp(input[i] - max_val); // Subtract max for stability
        sum += output_dest[i];
    }
    for (int i = 0; i < size; i++) output_dest[i] /= sum;
}


/* Forward pass kernel for a layer */
__global__ void forward_kernel(double *input, double *weights, double *biases,
                          double *preactivation, double *activation, int input_size, int output_size) {
    // determing threads to calculate each neuron
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Only work if this thread is corresponding to a valid neuron
    if (neuron_idx < output_size) {
        double sum = biases[neuron_idx];

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
__global__ void zero_layer_gradients(double *grad_b, double *grad_in, int num_nodes) {
    int output_dim_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (output_dim_idx < num_nodes) {
        grad_b[output_dim_idx] = 0;
        grad_in[output_dim_idx] = 0;
    }
}

// Dedicated Kernel for Weight Gradients (Size: input_size * output_size)
__global__ void zero_weight_gradients(double *grad_w, int total_weights) {
    int weight_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (weight_idx < total_weights) {
        grad_w[weight_idx] = 0.0;
    }
}

__global__ void backward_kernel_output_layer(
    double* grad_wrt_b, 
    double* grad_wrt_w, 
    double* previous_grad_wrt_input,
    double* previous_activations,
    double* weights,
    int input_size, 
    int output_size, 
    int target, 
    double *probs) 
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < output_size) {
        double target_val = (j == target) ? 1.0 : 0.0;
        double delta = probs[j] - target_val;

        grad_wrt_b[j] = delta;

        for (int k = 0; k < input_size; k++) {
            int weight_idx = k * output_size + j;
            grad_wrt_w[weight_idx] = delta * previous_activations[k];
            // Use atomicAdd to prevent race conditions from multiple 'j' threads
            atomicAdd(&previous_grad_wrt_input[k], delta * weights[weight_idx]);
        }
    }
}

__global__ void backward_kernel(Layer* layer, Layer* previous_layer, double* initial_inputs, int current_layer_idx) {
    
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (neuron_idx < layer->output_size) {
        double delta = 0;

        double relu_derivation = layer->activations[neuron_idx] == 0 ? 0.0 : 1.0; // If relu is not 0 then gradient is 1
        delta = layer->grad_wrt_input[neuron_idx] * relu_derivation;

        layer->grad_wrt_b[neuron_idx] = delta;

        for ( int input_size_idx = 0; input_size_idx < layer->input_size; input_size_idx++ ) {
            // weights are [input_size * output_size]
            // Accessing weights for input j and neuron j:
            int weight_idx = input_size_idx * layer->output_size + neuron_idx;

            // Gradient is delta * activation of the previous layer
            double *prev_act = (current_layer_idx == 0) ? initial_inputs : previous_layer->activations;
            layer->grad_wrt_w[weight_idx] = delta * prev_act[input_size_idx];

            // Pass error back to the previous layer
            if (current_layer_idx != 0) {
                previous_layer->grad_wrt_input[input_size_idx] += delta * layer->weights[weight_idx];
            }
        }
    }
}

__global__ void update_kernel(double* weights, double* biases, double* grad_w, double* grad_b, 
                             int in_size, int out_size, double learning_rate) {
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

void forward(Network *network, double *initial_inputs) {

    int threadsPerBlock = 256;
    
    double *current_input = initial_inputs;
    // calculate the forward pass for all nodes in the network
    for (int x = 0; x < network->num_layers; x++) {

        // loop over layers in network
        Layer* layer = &network->layers[x];

        int blocksPerGrid = (layer->output_size + threadsPerBlock -1) / threadsPerBlock;

        // Launch kernel for forward pass
        forward_kernel <<<blocksPerGrid, threadsPerBlock>>>(current_input, layer->weights, layer->biases,
                                                            layer->preactivations, layer->activations,
                                                            layer->input_size, layer->output_size);

        // change the intput to the next layer
        current_input = layer->activations;

        // Wait for the GPU to finish before moving to the next layer
        cudaDeviceSynchronize();
    }
}

void backward(Network *network, int target, double* initial_inputs, double *probs) {
    int threadsPerBlock = 256;

    // 1. Zero all gradients first
    for (int i = 0; i < network->num_layers; i++) {
        Layer *l = &network->layers[i];
        int b_zero = (l->output_size + threadsPerBlock - 1) / threadsPerBlock;
        int w_zero = (l->input_size * l->output_size + threadsPerBlock - 1) / threadsPerBlock;
        
        zero_layer_gradients<<<b_zero, threadsPerBlock>>>(l->grad_wrt_b, l->grad_wrt_input, l->output_size);
        zero_weight_gradients<<<w_zero, threadsPerBlock>>>(l->grad_wrt_w, l->input_size * l->output_size);
    }
    cudaDeviceSynchronize();

    // 2. Output Layer
    Layer* out_l = &network->layers[network->num_layers - 1];
    Layer* prev_l = &network->layers[network->num_layers - 2];
    
    for (int i = 0; i < 10; i++) {
        // zero probs
        probs[i] = 0;
    }
    int blocks = (out_l->output_size + threadsPerBlock - 1) / threadsPerBlock;
    int num_classes = 10;
    softmax_kernel<<<blocks, threadsPerBlock>>>(network->layers[network->num_layers-1].activations, num_classes, probs);

    backward_kernel_output_layer<<<blocks, threadsPerBlock>>>(
        out_l->grad_wrt_b, out_l->grad_wrt_w, prev_l->grad_wrt_input, 
        prev_l->activations, out_l->weights, 
        out_l->input_size, out_l->output_size, target, probs
    );

    // 3. Hidden Layers (Loop from pen-ultimate layer down to 0)
    for (int i = network->num_layers - 2; i >= 0; i--) {
        Layer *layer = &network->layers[i];
        Layer *previous_layer = (i > 0) ? &network->layers[i - 1] : NULL;
        
        // Launch backward Kernel Jalla Jalla
        blocks = (layer->output_size + threadsPerBlock -1) / threadsPerBlock;
        backward_kernel<<<blocks, threadsPerBlock>>>(layer, previous_layer, initial_inputs, i);

        cudaDeviceSynchronize();
    }
}
 

/**
 * Function for initializing the the MLP layers on the GPU
 */
void init_layer(Layer *l, int in, int out) {
    l->input_size = in; // input dimetion
    l->output_size = out; // output dimetion

    // Allocate memory using cudaMalloc
    cudaMalloc((void**)&l->weights, sizeof(double) * in * out);
    cudaMalloc((void**)&l->biases, sizeof(double) * out);
    cudaMalloc((void**)&l->preactivations, sizeof(double) * out);
    cudaMalloc((void**)&l->activations, sizeof(double) * out);
    cudaMalloc((void**)&l->grad_wrt_w, sizeof(double) * out * in);
    cudaMalloc((void**)&l->grad_wrt_b, sizeof(double) * out);
    cudaMalloc((void**)&l->grad_wrt_input, sizeof(double) * in);

    // allocate temp weights to cpu
    double *temp_weights = (double*) malloc(in * out * sizeof(double));
    // Using HE initialization for ReLU activations
    double scale = sqrt(2.0 / in);
    for (int i = 0; i < in * out; i++) {
        // initialize values in the range of -1 to 1;
        temp_weights[i] = (((double) rand() / RAND_MAX) * 2.0 - 1.0) * scale;
    }

    double *temp_biases = (double*) malloc(out * sizeof(double));
    for (int i = 0; i < out; i++) {
        temp_biases[i] = 1.0f;
    }

    // Copy the weights and biases from the cpu to the gpu
    cudaMemcpy(l->weights, temp_weights, in * out * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(l->biases, temp_biases, out * sizeof(double), cudaMemcpyHostToDevice);
    // free the temp weights from cpu ram
    free(temp_weights);
    free(temp_biases);

}


Network create_network_layers(int* layer_sizes, int num_layers) {

    Layer* layers;
    cudaMallocManaged((void**)&layers, num_layers * sizeof(Layer));
    
    for (int i = 0; i < num_layers; i++) {
        int in_size = layer_sizes[i];
        int out_size = layer_sizes[i+1];
        init_layer(&layers[i], in_size, out_size);
    }

    Network network = {
        .layers = layers,
        .num_layers = num_layers
    };

    return network;
}

void update(Network *network, double learning_rate) {
    int threadsPerBlock = 256;
    for (int i = 0; i < network->num_layers; i++) {
        Layer *l = &network->layers[i];
        int blocks = (l->output_size + threadsPerBlock - 1) / threadsPerBlock;

        update_kernel<<<blocks, threadsPerBlock>>>(
            l->weights, l->biases, l->grad_wrt_w, l->grad_wrt_b, 
            l->input_size, l->output_size, learning_rate
        );
    }
    cudaDeviceSynchronize(); 
}


// Function the get the max value index after applying softmax in the network
int get_max_value_index(double* list, int size) {
    double max = -1;
    int idx = -1;
    for (int i = 0; i < size; i++) {
        double value = list[i];
        if (value > max) {
            max = value;
            idx = i;
        }
    }
    return idx;
}


/*
** This is my recreational/passion machine learning project in C
**
** Author: Aaro Karhu
*/
int main() {

    srand(time(NULL));

    init_mnist_buffers(); // Allocate the 1D arrays

    // Load data
    printf("Loading image data\n");
    read_mnist_char(TRAIN_IMAGE, NUM_TRAIN, SIZE, train_image_char);
    read_mnist_char(TEST_IMAGE, NUM_TEST, SIZE, test_image_char);
    read_mnist_char(TRAIN_LABEL, NUM_TRAIN, 1, train_label_char);
    read_mnist_char(TEST_LABEL, NUM_TEST, 1, test_label_char);

    // Convert to double
    printf("Parsing image data to vector values\n");
    image_char2double(NUM_TRAIN, train_image_char, train_image);
    image_char2double(NUM_TEST, test_image_char, test_image);
    label_char2int(NUM_TRAIN, train_label_char, train_image_label);
    label_char2int(NUM_TEST, test_label_char, test_image_label);

    // NOW you can easily copy to GPU
    printf("Allocating memory for the gpu for images\n");
    double *d_train_image;
    cudaMalloc((void**)&d_train_image, NUM_TRAIN * SIZE * sizeof(double));
    cudaMemcpy(d_train_image, train_image, NUM_TRAIN * SIZE * sizeof(double), cudaMemcpyHostToDevice);

    printf("Initializing network\n");
    int layer_sizes[] = {784, 128, 64, 10};
    int num_layers = LEN(layer_sizes) - 1;
    int num_classes = 10;
    Network network = create_network_layers(layer_sizes, num_layers);

    // Allocate probability array (array after softmax) for the host and device
    double *dh_probs;
    cudaMallocManaged((void**)&dh_probs, sizeof(double) * num_classes);
    
    // allocate memory on the device (GPU) for inputs
    double *d_input;
    cudaMalloc((void**)&d_input, sizeof(double) * 784);

    // allocate memory on the device for target valeus
    double *d_target;
    cudaMalloc((void**)&d_target, sizeof(double));
    int epochs = 20;
    double learning_rate = 0.01;

    printf("Started training\n");
    for (int e = 0; e < epochs; e++) {

        double prediction_precent = 0;
        int correct_predictions = 0;
        double epoch_loss = 0.0;

        int N = 60000;
        for (int i = 0; i < N; i++) {
            // 1. COPY INPUT: Move current image data to GPU
            cudaMemcpy(d_input, train_image + (i * 784), sizeof(double) * 784, cudaMemcpyHostToDevice);
            cudaMemcpy(d_target, train_image_label + i, sizeof(int), cudaMemcpyHostToDevice);

            // 2. FORWARD: Pass the GPU buffer to your forward function
            forward(&network, d_input);

            // 3. SOFTMAX: Calculate the softmax in the GPU
            double *output_layer_activations_ptr = network.layers[network.num_layers -1].activations;
            softmax_kernel<<<1, 1>>>(output_layer_activations_ptr, num_classes, dh_probs);

            cudaDeviceSynchronize();

            int prediction = get_max_value_index(dh_probs, num_classes);
            if (prediction == train_image_label[i]) correct_predictions++;
            // printf("%d %d\n", prediction, train_image_label[i]);
            
            // 4. LOSS: Calculate the loss of the network
            epoch_loss += sparce_categorical_cross_entropy(dh_probs, train_image_label[i]);

            // 5. BACKWARD: Zero gradients and calculate gradients
            backward(&network, train_image_label[i], d_input, dh_probs);

            // 6. UPDATE PARAMETERS
            update(&network, learning_rate);
        }
        prediction_precent = 100 - (double)correct_predictions / N * 100;
        printf("Epoch %d | Avg Loss: %.4f | train error: %.2f% \n", e+1, epoch_loss / N, prediction_precent);
    }


    // Free all of host and device memory

    for (int i = 0; i < network.num_layers; i++) {
        cudaFree(network.layers[i].activations);
        cudaFree(network.layers[i].preactivations);
        cudaFree(network.layers[i].biases);
        cudaFree(network.layers[i].weights);
        cudaFree(network.layers[i].grad_wrt_b);
        cudaFree(network.layers[i].grad_wrt_w);
        cudaFree(network.layers[i].grad_wrt_input);
    }

    cudaFree(network.layers); // Use cudaFree for Managed memory!

    cudaFree(d_input);
    cudaFree(dh_probs);
    cudaFree(d_train_image);
    free(train_image);
    free(test_image);
    free(train_image_label);
    free(test_image_label);
    free(train_image_char);
    free(test_image_char);
    free(train_label_char);
    free(test_label_char);

    cudaDeviceReset();

    return 0;
}
