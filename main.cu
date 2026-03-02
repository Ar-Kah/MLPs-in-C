#include <string.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

#include "mnist.h"

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

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double activation) {
    return activation * (1.0f -activation);
}

// Implementation of softmax layer
void apply_softmax(double* input, int size, double* output_dest) {
    double max_val = input[0];
    for(int i = 1; i < size; i++) if(input[i] > max_val) max_val = input[i];

    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        output_dest[i] = exp(input[i] - max_val); // Subtract max for stability
        sum += output_dest[i];
    }
    for (int i = 0; i < size; i++) output_dest[i] /= sum;
}
// relu layer
double relu(double output) {
    double result;
    double numerator = output + fabs(output);
    result = numerator / 2;
    return result;
}


// =====================
// Loss functions      #
// =====================

double bce(double y, double t) {
    if (t <= 0 || t >= 1) {
        // Avoid log(0) or log(1) issues; clit for numerical stability
        t = (t <= 0) ? 1e-7 : (t >= 1) ? 1 - 1e-7 : t;
    }
    return - (y * log(t) + (1 - y) * log(1 - t));
}

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

__global__ void forward_kernel(double *input, double *weights, double *biases,
                          double *preactivation, double *activation, int input_size, int output_size) {
    // determing threads to calculate each neuron
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Only work if this thread is corresponding to a valid neuron
    if (j < output_size) {
        double sum = biases[j];

        for (int i = 0; i < input_size; i++) {
            sum += input[i] * weights[i * output_size + j];
        }
        preactivation[j] = sum; // store the preactivation for backward pass
        activation[j] = sum > 0 ? sum : 0; // ReLU activation
    }
}

void forward(Network *network, double *initial_inputs) {
    
    double *current_input = initial_inputs;
    // calculate the forward pass for all nodes in the network
    for (int x = 0; x < network->num_layers; x++) {

        // loop over layers in network
        Layer* layer = &network->layers[x];

        // Launch a forward kernel for MLP calculations inside a layer
        int threadsPerBlock = 256;
        int blocksPerGrid = (layer->output_size + threadsPerBlock -1) / threadsPerBlock;

        forward_kernel <<<blocksPerGrid, threadsPerBlock>>>(current_input, layer->weights, layer->biases,
                                                            layer->preactivations, layer->activations, layer->input_size, layer->output_size);

        current_input = layer->activations;

        // Wait for the GPU to finish before moving to the next layer
        cudaDeviceSynchronize();
    }
}

void backward(Network *network, int target, double* initial_inputs, char* actication) {
    // 1. Reset all grad_wrt_input to 0 so we can accumulate sums
    for (int i = 0; i < network->num_layers; i++) {
        for (int j = 0; j < network->layers[i].input_size; j++) {
            network->layers[i].grad_wrt_input[j] = 0.0f;
        }
    }

    // 2. Loop backwards through layers
    for (int i = network->num_layers - 1; i >= 0; i--) {
        Layer *layer = &network->layers[i];
        
        // Determine what the inputs to THIS layer were
        double *layer_inputs = (i == 0) ? initial_inputs : network->layers[i-1].activations;

        for (int j = 0; j < layer->output_size; j++) {
            double delta;

            // Edge case for the last layer in the network
            // Gradinet is calculated wrt. loss function
            if (i == network->num_layers - 1) {
                // Correct Softmax + Cross-Entropy gradient simplyfies to prob - target
                double probs[10]; // hardcode target class amount
                apply_softmax(layer->activations, layer->output_size, probs);

                // Create one-hot target on the fly
                double target_val = (j == target) ? 1.0 : 0.0;
                delta = probs[j] - target_val; 
            }
            else {
                // HIDDEN LAYER: (Error from layer above) * derivative
                if (strcmp(actication, "sigmoid") == 0) {
                    delta = layer->grad_wrt_input[j] * sigmoid_derivative(layer->activations[j]);
                }
                else {
                    // derivative of relu activation
                    double relu_derivative = (layer->activations[j] > 0) ? 1.0 : 0.0;
                    delta = layer->grad_wrt_input[j] * relu_derivative;
                }
            }

            // skip further gradient calculations if delta is 0
            /* if (delta == 0) continue; */

            // Bias gradient is just the delta
            layer->grad_wrt_b[j] = delta;

            // WEIGHT GRADIENTS: Loop over every input that fed into this neuron
            for (int k = 0; k < layer->input_size; k++) {
                int weight_idx = k * layer->output_size + j;
                
                // Weight gradient = delta * input
                layer->grad_wrt_w[weight_idx] = delta * layer_inputs[k];

                // PASS ERROR BACK: Update the input gradient for the PREVIOUS layer
                // We sum these up because one input node affects multiple output nodes
                if (i > 0) {
                    network->layers[i-1].grad_wrt_input[k] += delta * layer->weights[weight_idx];
                }
            }
        }
    }
    // uncomment to check vanishing gradients
    /* printf("%f\n", network->layers[network->num_layers -1].grad_wrt_w[0]); */
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
    double *temp_weights = (double*)malloc(in * out * sizeof(double));
    // Using HE initialization for ReLU activations
    double scale = sqrt(2.0 / in);
    for (int i = 0; i < in * out; i++) {
        // initialize values in the range of -1 to 1;
        temp_weights[i] = (((double) rand() / RAND_MAX) * 2.0 - 1.0) * scale;
    }

    double *temp_biases = (double*)malloc(out * sizeof(double));
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

// Inside loss()
double* loss(Layer output_layer, double* targets) {
    double *losses;
    cudaMalloc((void**)&losses, sizeof(double) * output_layer.output_size);
    for (int i = 0; i < output_layer.output_size; i++) {
        losses[i] = bce(targets[i], output_layer.activations[i]);
        printf("Loss for output %d: %f (Pred: %f, Target: %f)\n", 
                i, losses[i], output_layer.activations[i], targets[i]);
    }
    return losses;
}

Network create_network_layers(int* layer_sizes, int num_layers) {

    Layer* layers = (Layer*)malloc(sizeof(Layer) * num_layers);
    
    for (int i = 0; i < num_layers; i++) {
        int in_size = layer_sizes[i];
        int out_size = layer_sizes[i+1];
        init_layer(&layers[i], in_size, out_size);
    }

    Network network = { .layers = layers, .num_layers = num_layers };
    return network;
}


void update(Network *network, double learning_rate) {
    for (int i = 0; i < network->num_layers; i++) {
        Layer *layer = &network->layers[i];

        for (int j = 0; j < layer->output_size; j++) {
            // Update Bias
            layer->biases[j] -= learning_rate * layer->grad_wrt_b[j];

            for (int k = 0; k < layer->input_size; k++) {
                int idx = k * layer->output_size + j;
                // update weights
                layer->weights[idx] -= learning_rate * layer->grad_wrt_w[idx];
            }
        }
    }
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
    read_mnist_char(TRAIN_IMAGE, NUM_TRAIN, SIZE, train_image_char);
    read_mnist_char(TEST_IMAGE, NUM_TEST, SIZE, test_image_char);
    read_mnist_char(TRAIN_LABEL, NUM_TRAIN, 1, train_label_char);
    read_mnist_char(TEST_LABEL, NUM_TEST, 1, test_label_char);

    // Convert to double
    image_char2double(NUM_TRAIN, train_image_char, train_image);
    image_char2double(NUM_TEST, test_image_char, test_image);

    // NOW you can easily copy to GPU
    double *d_train_image;
    cudaMalloc(&d_train_image, NUM_TRAIN * SIZE * sizeof(double));
    cudaMemcpy(d_train_image, train_image, NUM_TRAIN * SIZE * sizeof(double), cudaMemcpyHostToDevice);

    int layer_sizes[] = {784, 128, 64, 10};
    int num_layers = LEN(layer_sizes) - 1;
    Network network = create_network_layers(layer_sizes, num_layers);

    // FIX: Correct cudaMalloc usage
    double *d_probs;
    cudaMalloc((void**)&d_probs, sizeof(double) * 10);
    
    // Create a buffer on GPU for the input image
    double *d_input;
    cudaMalloc((void**)&d_input, sizeof(double) * 784);

    int epochs = 20;

    for (int e = 0; e < epochs; e++) {
        for (int i = 0; i < 60000; i++) {
            // 1. COPY INPUT: Move current image to GPU
            cudaMemcpy(d_input, train_image + (i * 784), sizeof(double) * 784, cudaMemcpyHostToDevice);

            // 2. FORWARD: Pass the GPU buffer to your forward function
            // (You will need to update your forward function to accept d_input)
            forward(&network, d_input);

            // 3. BACKWARD/UPDATE: 
            // Warning: These are currently CPU functions. To use them, you must 
            // use cudaMemcpy to bring layer->activations back to the CPU first.
            // ... (Your current code here will crash unless you bring data back)
            
        }
    }
    return 0;
}
