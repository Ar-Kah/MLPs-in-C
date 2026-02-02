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

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
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

int main() {
    float inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    float targets[4][1] = {{0}, {1}, {1}, {0}};
    srand(time(NULL));
    // layer initialization
    Layer layer = {0};
    Layer *lp = &layer;
    int input_dimetion = 2;
    int output_dimention = 1;
    init_layer(lp, input_dimetion, output_dimention);
    forward(lp, *inputs);
    printf("Prediction: %f\n", output_layer.activations[0]);
    return 0;
}
