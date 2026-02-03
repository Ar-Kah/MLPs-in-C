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

float BCE(float y, float t) {
    if (t <= 0 || t >= 1) {
        // Avoid log(0) or log(1) issues; clit for numerical stability
        t = (t <= 0) ? 1e-7 : (t >= 1) ? 1 - 1e-7 : t;
    }
    return - (y * log(t) + (1 - y) * log(1 - t));
}

void forward(Layer* layer, float* inputs) {
    for (int j = 0; j < layer->output_size; j++) {
        float sum = layer->biases[j];
        for (int i = 0; i < layer->input_size; i++) {
            sum += inputs[i] * layer->weights[i * layer->output_size + j];
        }
    layer->preactivations[j] = sum;
    layer->activations[j] = sigmoid(sum);
    }
}

void backward() {
    
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
        l->weights[i] = ((float) rand() / (float)RAND_MAX) * 2.0 - 1.0; // initialize values in the range of -1 to 1
    }

    for (int i = 0; i < out; i++) {
        l->biases[i] = 0.01f;
    }
}

float* loss(float* outputs, float* targets) {
    int length = sizeof(&outputs) / sizeof(outputs[0]);
    float *losses = malloc(sizeof(float) * length);
    for (int i = 0; i < 4; i++) {
        losses[i] = BCE(outputs[i], targets[i]);
        printf("%f\n", losses[i]);
    }
    return losses;
}

int main() {
    float inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    float targets[4][1] = {{0}, {1}, {1}, {0}};
    srand(time(NULL));
    // layer initialization
    Layer layer1 = {0};
    init_layer(&layer1, 2, 3);

    Layer output_layer = {0};
    init_layer(&output_layer, 3, 1);
    
    forward(&layer1, *inputs);
    forward(&output_layer, layer1.activations);

    float* losses = loss(output_layer.activations, *targets);

    return 0;
}
