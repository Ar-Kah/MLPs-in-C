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
double *relu(double *outputs) {
    int num_outputs = sizeof(&outputs) / sizeof(outputs[0]);
    double* result = malloc(sizeof(double) * num_outputs);
    for (int i = 0; i < num_outputs; i++) {
        double numerator = outputs[i] + fabs(outputs[i]);
        result[i] = numerator / 2;
    }
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

void forward(Network *network, double *initial_inputs) {
    
    double *current_input = initial_inputs;
    // calculate the forward pass for all nodes in the network
    for (int x = 0; x < network->num_layers; x++) {

        // loop over layers in network
        Layer* layer = &network->layers[x];

        for (int j = 0; j < layer->output_size; j++) {

            // calculate mlp forwardpass calculations
            double sum = layer->biases[j];

            for (int i = 0; i < layer->input_size; i++) {
                sum += current_input[i] * layer->weights[i * layer->output_size + j];
            }

        // store claculations into activation and preactivation 
        layer->preactivations[j] = sum;
        layer->activations[j] = sigmoid(sum);
        }
    current_input = layer->activations;
    }
}

void backward(Network *network, int target, double* initial_inputs) {
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

            if (i == network->num_layers - 1) {
                // Correct Softmax + Cross-Entropy gradient
                double probs[10];
                apply_softmax(layer->activations, layer->output_size, probs);

                // Create one-hot target on the fly
                double target_val = (j == target) ? 1.0 : 0.0;
                delta = probs[j] - target_val; 
            }
            else {
                // HIDDEN LAYER: (Error from layer above) * derivative
                delta = layer->grad_wrt_input[j] * sigmoid_derivative(layer->activations[j]);
            }

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
}

void init_layer(Layer *l, int in, int out) {
    l->input_size = in; // input dimetion
    l->output_size = out; // output dimetion

    l->weights = malloc(sizeof(double) * in * out);
    l->biases = malloc(sizeof(double) * out);

    l->preactivations = malloc(sizeof(double) * out);
    l->activations = malloc(sizeof(double) * out);

    l->grad_wrt_w = malloc(sizeof(double) * out *in);
    l->grad_wrt_b = malloc(sizeof(double) * out);
    l->grad_wrt_input = malloc(sizeof(double) * in);

    // randomly initialize the weights and biases
    for (int i = 0; i < in * out; i++) {
        // initialize values in the range of -1 to 1;
        l->weights[i] = ((double) rand() / (double)RAND_MAX) * 2.0 - 1.0;
    }

    for (int i = 0; i < out; i++) {
        l->biases[i] = 0.01f;
    }
}

// Inside loss()
double* loss(Layer output_layer, double* targets) {
    double *losses = malloc(sizeof(double) * output_layer.output_size);
    for (int i = 0; i < output_layer.output_size; i++) {
        losses[i] = bce(targets[i], output_layer.activations[i]);
        printf("Loss for output %d: %f (Pred: %f, Target: %f)\n", 
                i, losses[i], output_layer.activations[i], targets[i]);
    }
    return losses;
}

Network create_network_layers(int* layer_sizes, int num_layers) {

    Layer* layers = malloc(sizeof(Layer) * num_layers);
    
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


/*
** This is my recreational/passion machine learning project in C
**
** Author: Aaro Karhu
*/
int main() {
    srand(time(NULL)); // initialize random num generator

    // load mnist data
    load_mnist();

    /* double inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}}; */
    /* double targets[4][1] = {{0}, {1}, {1}, {0}}; */
    /* int layer_sizes[] = {2, 3, 1}; */


    // define the dense network for the mnist dataset
    int layer_sizes[] = {784, 128, 64, 10}; // 784 input 10 output

    int num_layers = LEN(layer_sizes) -1;

    Network network = create_network_layers(layer_sizes, num_layers);

    double learning_rate = 0.1f;
    int epochs = 20;

    int num_classes = layer_sizes[num_layers]; // get the number of classes from
    Layer last_layer = network.layers[num_layers - 1]; // get the last layer

    double* probs = malloc(sizeof(double) * num_classes);

    for (int e = 0; e < epochs; e++) {
        double epoch_loss = 0;
        // i < 4 is just for testing; MNIST train size is usually 60,000
        for (int i = 0; i < 600; i++) { 
            forward(&network, train_image[i]);

            Layer *last_layer_ptr = &network.layers[num_layers - 1];
            apply_softmax(last_layer_ptr->activations, num_classes, probs);

            epoch_loss += sparce_categorical_cross_entropy(probs, train_label[i]);

            // Pass the target label (0-9)
            backward(&network, train_label[i], train_image[i]);

            update(&network, learning_rate);
        }
        printf("Epoch %d | Avg Loss: %f\n", e, epoch_loss / 600.0f);
    }
    free(probs);
    return 0;
}
