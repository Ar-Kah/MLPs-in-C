#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    int input_size;
    int output_size;
    float *weights;
    float *biases;
    float *preactivations;
    float *activations;
    float *grad_wrt_w;
    float *grad_wrt_b;
    float *grad_wrt_input;
} Layer;

typedef struct {
    Layer *layers;
    int num_layers;
} Network;

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

float sigmoid_derivative(float activation) {
    return activation * (1.0f -activation);
}

float bce(float y, float t) {
    if (t <= 0 || t >= 1) {
        // Avoid log(0) or log(1) issues; clit for numerical stability
        t = (t <= 0) ? 1e-7 : (t >= 1) ? 1 - 1e-7 : t;
    }
    return - (y * log(t) + (1 - y) * log(1 - t));
}

// The derivative of BCE Loss with respect to the Sigmoid activation
float bce_derivative(float target, float prediction) {
    // We add a tiny epsilon to prevent division by zero
    float epsilon = 1e-7f;
    if (prediction < epsilon) prediction = epsilon;
    if (prediction > 1.0f - epsilon) prediction = 1.0f - epsilon;

    return (prediction - target) / (prediction * (1.0f - prediction));
}

void forward(Network *network, float* initial_inputs) {
    
    float* current_input = initial_inputs;
    // calculate the forward pass for all nodes in the network
    for (int x = 0; x < network->num_layers; x++) {

        // loop over layers in network
        Layer* layer = &network->layers[x];

        for (int j = 0; j < layer->output_size; j++) {

            // calculate mlp forwardpass calculations
            float sum = layer->biases[j];

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

void backward(Network *network, float* targets, float* initial_inputs) {
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
        float *layer_inputs = (i == 0) ? initial_inputs : network->layers[i-1].activations;

        for (int j = 0; j < layer->output_size; j++) {
            float delta;

            if (i == network->num_layers - 1) {
                // OUTPUT LAYER: (Prediction - Target)
                delta = layer->activations[j] - targets[j];
            } else {
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

    l->weights = malloc(sizeof(float) * in * out);
    l->biases = malloc(sizeof(float) * out);

    l->preactivations = malloc(sizeof(float) * out);
    l->activations = malloc(sizeof(float) * out);

    l->grad_wrt_w = malloc(sizeof(float) * out *in);
    l->grad_wrt_b = malloc(sizeof(float) * out);
    l->grad_wrt_input = malloc(sizeof(float) * in);

    // randomly initialize the weights and biases
    for (int i = 0; i < in * out; i++) {
        // initialize values in the range of -1 to 1;
        l->weights[i] = ((float) rand() / (float)RAND_MAX) * 2.0 - 1.0;
    }

    for (int i = 0; i < out; i++) {
        l->biases[i] = 0.01f;
    }
}

// Inside loss()
float* loss(Layer output_layer, float* targets) {
    float *losses = malloc(sizeof(float) * output_layer.output_size);
    for (int i = 0; i < output_layer.output_size; i++) {
        losses[i] = bce(targets[i], output_layer.activations[i]);
        printf("Loss for output %d: %f (Pred: %f, Target: %f)\n", 
                i, losses[i], output_layer.activations[i], targets[i]);
    }
    return losses;
}

Layer* create_network_layers(int num_layers) {
    // Allocate memory for the layer structures themselves on the heap
    Layer* layers = malloc(sizeof(Layer) * num_layers);
    
    // Initialize Layer 0: 2 inputs -> 3 outputs
    init_layer(&layers[0], 2, 3);
    // Initialize Layer 1: 3 inputs -> 1 output
    init_layer(&layers[1], 3, 1);
    
    return layers;
}


void update(Network *network, float learning_rate) {
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

int main() {
    float inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    float targets[4][1] = {{0}, {1}, {1}, {0}};
    srand(time(NULL));

    int num_layers = 2;
    Layer* layers = create_network_layers(num_layers);
    Network network = { .layers = layers, .num_layers = num_layers };

    float learning_rate = 0.1f;
    int epochs = 20000;

    for (int e = 0; e < epochs; e++) {
        float epoch_loss = 0;

        for (int i = 0; i < 4; i++) {
            // 1. Forward pass with ONE of the four inputs
            forward(&network, inputs[i]);

            // 2. Track loss
            epoch_loss += bce(targets[i][0], network.layers[1].activations[0]);

            // 3. Backward pass
            backward(&network, targets[i], inputs[i]);

            // 4. Update
            update(&network, learning_rate);
        }

        if (e % 2000 == 0) {
            printf("Epoch %d | Avg Loss: %f\n", e, epoch_loss / 4.0f);
        }
    }

    // Test it!
    printf("\nFinal Predictions:\n");
    for(int i = 0; i < 4; i++) {
        forward(&network, inputs[i]);
        printf("In: [%.0f, %.0f] Target: [%.0f] Pred: %f\n", 
                inputs[i][0], inputs[i][1], targets[i][0], network.layers[1].activations[0]);
    }

    return 0;
}
