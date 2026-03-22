#include <string.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

#include "kernels.cuh"
#include "mnist.h"

// define macro for getting the size of a list of integers
#define LEN(x) (sizeof(x) / sizeof(x[0]))

typedef struct {
    Layer *layers;
    int num_layers;
} Network;


/* After taking the highest index after applying softmax
 * all but one index is 0 so we only calculate the loss
 * for that index */
float sparce_categorical_cross_entropy(float *predictions, int label_index) {
    // prediction is the array from softmax
    // label_index is the actual number of 0-9
    float p = predictions[label_index];

    if (p < 1e-15) p = 1e-15;

    return -log(p);
}

// The derivative of BCE Loss with respect to the Sigmoid activation
float bce_derivative(float target, float prediction) {
    // We add a tiny epsilon to prevent division by zero
    float epsilon = 1e-7f;
    if (prediction < epsilon) prediction = epsilon;
    if (prediction > 1.0f - epsilon) prediction = 1.0f - epsilon;

    return (prediction - target) / (prediction * (1.0f - prediction));
}


void forward(Network *network, float *initial_inputs, int batch_size) {

    dim3 threadsPerBlock(16, 16);
    
    float *current_input = initial_inputs;
    // calculate the forward pass for all nodes in the network
    for (int x = 0; x < network->num_layers; x++) {

        // loop over layers in network
        Layer* layer = &network->layers[x];

        dim3 blocksPerGrid(
            (layer->output_size + threadsPerBlock.x -1) / threadsPerBlock.x,
            (batch_size + threadsPerBlock.y -1) / threadsPerBlock.y
        );

        // Launch kernel for forward pass
        forward_kernel2d <<<blocksPerGrid, threadsPerBlock>>>(current_input, layer->weights, layer->biases,
                                                            layer->preactivations, layer->activations,
                                                            layer->input_size, layer->output_size, batch_size);

        // change the intput to the next layer
        current_input = layer->activations;

        // Wait for the GPU to finish before moving to the next layer
        CUDA_CHECK_ERR(cudaDeviceSynchronize());
    }
}

void backward(Network *network, int *target, float* initial_inputs, float *probs, int batch_size) {

    dim3 threadsPerBlock(16, 16);

    // 1. Zero all gradients first
    for (int i = 0; i < network->num_layers; i++) {
        Layer *l = &network->layers[i];

        // cudaMemset for makeing sure that there is no gradient leak
        cudaMemset(l->grad_wrt_b, 0, sizeof(float) * l->output_size);
        cudaMemset(l->grad_wrt_w, 0, sizeof(float) * l->input_size * l->output_size);
        cudaMemset(l->grad_wrt_input, 0, sizeof(float) * l->input_size * batch_size);
    }
    CUDA_CHECK_ERR(cudaDeviceSynchronize());

    // 2. Output Layer
    Layer* out_l = &network->layers[network->num_layers - 1];
    Layer* prev_l = &network->layers[network->num_layers - 2];
    
    dim3 blocksPerGrid(
        (out_l->output_size + threadsPerBlock.x -1) / threadsPerBlock.x,
        (batch_size + threadsPerBlock.y -1) / threadsPerBlock.y
    );
    
    backward_kernel_output_layer2d<<<blocksPerGrid, threadsPerBlock>>>(
        out_l->grad_wrt_b, out_l->grad_wrt_w, prev_l->grad_wrt_input, 
        prev_l->activations, out_l->weights, 
        out_l->input_size, out_l->output_size, batch_size, target, probs
    );

    // 3. Hidden Layers (Loop from pen-ultimate layer down to 0)
    for (int i = network->num_layers - 2; i >= 0; i--) {
        Layer *layer = &network->layers[i];
        Layer *previous_layer = (i > 0) ? &network->layers[i - 1] : NULL;
        
        // Launch backward Kernel Jalla Jalla
        blocksPerGrid = dim3(
                (layer->output_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y
            );

        backward_kernel2d <<<blocksPerGrid, threadsPerBlock>>> (layer, previous_layer, initial_inputs,
                                                                i, batch_size);
        CUDA_CHECK_ERR(cudaDeviceSynchronize());
    }
}
 

/**
 * Function for initializing the the MLP layers on the GPU
 */
void init_layer(Layer *l, int in, int out, int batch_size) {
    l->input_size = in; // input dimetion
    l->output_size = out; // output dimetion

    // Allocate memory using cudaMalloc
    cudaMalloc((void**)&l->weights, sizeof(float) * in * out);
    cudaMalloc((void**)&l->biases, sizeof(float) * out);
    cudaMalloc((void**)&l->preactivations, sizeof(float) * out * batch_size);
    cudaMalloc((void**)&l->activations, sizeof(float) * out * batch_size);
    cudaMalloc((void**)&l->grad_wrt_w, sizeof(float) * out * in);
    cudaMalloc((void**)&l->grad_wrt_b, sizeof(float) * out);
    cudaMalloc((void**)&l->grad_wrt_input, sizeof(float) * in * batch_size);

    // allocate temp weights to cpu
    float *temp_weights = (float*) malloc(in * out * sizeof(float));
    // Using HE initialization for ReLU activations
    float scale = sqrt(2.0 / in);
    for (int i = 0; i < in * out; i++) {
        // initialize values in the range of -1 to 1;
        temp_weights[i] = (((float) rand() / RAND_MAX) * 2.0 - 1.0) * scale;
    }

    float *temp_biases = (float*) malloc(out * sizeof(float));
    for (int i = 0; i < out; i++) {
        temp_biases[i] = 1.0f;
    }

    // Copy the weights and biases from the cpu to the gpu
    cudaMemcpy(l->weights, temp_weights, in * out * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l->biases, temp_biases, out * sizeof(float), cudaMemcpyHostToDevice);
    // free the temp weights from cpu ram
    free(temp_weights);
    free(temp_biases);

}

Network create_network_layers(int* layer_sizes, int num_layers, int batch_size) {

    Layer* layers;
    cudaMallocManaged((void**)&layers, num_layers * sizeof(Layer));
    
    for (int i = 0; i < num_layers; i++) {
        int in_size = layer_sizes[i];
        int out_size = layer_sizes[i+1];
        init_layer(&layers[i], in_size, out_size, batch_size);
    }

    Network network = {
        .layers = layers,
        .num_layers = num_layers
    };

    return network;
}

void update(Network *network, float learning_rate, int batch_size) {
    int threadsPerBlock = 256;
    for (int i = 0; i < network->num_layers; i++) {
        Layer *l = &network->layers[i];
        int blocks = (l->output_size + threadsPerBlock - 1) / threadsPerBlock;

        update_kernel_minibatch<<<blocks, threadsPerBlock>>>(
            l->weights, l->biases, l->grad_wrt_w, l->grad_wrt_b, 
            l->input_size, l->output_size, learning_rate, batch_size
        );
        CUDA_CHECK_ERR(cudaDeviceSynchronize()); 
    }
}

void update(Network *network, float learning_rate) {
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
int get_max_value_index(float* list, int size) {
    float max = -1;
    int idx = -1;
    for (int i = 0; i < size; i++) {
        float value = list[i];
        if (value > max) {
            max = value;
            idx = i;
        }
    }
    return idx;
}



void print_layer_stats(const char* name, float* d_grad, int size) {
    float *h_grad = (float*)malloc(sizeof(float) * size);
    cudaMemcpy(h_grad, d_grad, sizeof(float) * size, cudaMemcpyDeviceToHost);
    
    float sum = 0;
    for(int i = 0; i < size; i++) sum += fabs(h_grad[i]);
    
    printf("  [%s] Avg Grad Magnitude: %.10f\n", name, sum / size);
    free(h_grad);
}

/*
** This is my recreational/passion machine learning project in C
**
** Author: Aaro Karhu
*/
int main() {

    printf("Running test!\n");

    srand(time(NULL));

    init_mnist_buffers(); // Allocate the 1D arrays

    // Load data
    printf("Loading image data\n");
    read_mnist_char(TRAIN_IMAGE, NUM_TRAIN, SIZE, train_image_char);
    read_mnist_char(TEST_IMAGE, NUM_TEST, SIZE, test_image_char);
    read_mnist_char(TRAIN_LABEL, NUM_TRAIN, 1, train_label_char);
    read_mnist_char(TEST_LABEL, NUM_TEST, 1, test_label_char);

    // Convert to float
    printf("Parsing image data to vector values\n");
    image_char2float(NUM_TRAIN, train_image_char, train_image);
    image_char2float(NUM_TEST, test_image_char, test_image);
    label_char2int(NUM_TRAIN, train_label_char, train_image_label);
    label_char2int(NUM_TEST, test_label_char, test_image_label);

    // NOW you can easily copy to GPU
    printf("Allocating memory for the gpu for images\n");
    float *d_train_image;
    cudaMalloc((void**)&d_train_image, NUM_TRAIN * SIZE * sizeof(float));
    cudaMemcpy(d_train_image, train_image, NUM_TRAIN * SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize the batch size
    int batch_size = 128;

    printf("Initializing network\n");
    int layer_sizes[] = {784, 128, 64, 10};
    int num_layers = 3;
    int num_classes = 10;
    Network network = create_network_layers(layer_sizes, num_layers, batch_size);

    // Allocate probability array (array after softmax) for the host and device
    float *dh_probs;
    cudaMallocManaged((void**)&dh_probs, sizeof(float) * num_classes * batch_size);
    
    // allocate memory on the device (GPU) for inputs
    float *d_input;
    cudaMalloc((void**)&d_input, sizeof(float) * 784 * batch_size);

    // allocate memory on the device for target valeus
    int *d_target;
    cudaMalloc((void**)&d_target, sizeof(int) * batch_size);
    int epochs = 4;
    float learning_rate = 0.01;


    float *epoch_loss;
    cudaMallocManaged((void**)&epoch_loss, sizeof(float));

    printf("Started training\n");
    for (int e = 0; e < epochs; e++) {
        // CORRECTED: To zero a Managed Memory pointer, use the dereference operator
        *epoch_loss = 0.0; 

        int num_train_img = 60000;
        int step = 0;

        // Cut the loop before pointing at garbage memory in the taraining image array.
        // The loop should stop when we still have room for one full batch
        for (int i = 0; i + batch_size <= num_train_img; i += batch_size) {
            cudaMemcpy(d_input, train_image + (i * 784), sizeof(float) * 784 * batch_size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_target, train_image_label + i, sizeof(int) * batch_size, cudaMemcpyHostToDevice);

            forward(&network, d_input, batch_size);

            int threadsPerBlock = 256;
            int blocksPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
            
            float *output_layer_activations_ptr = network.layers[network.num_layers - 1].activations;
            safe_softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(output_layer_activations_ptr,
                                                               num_classes, dh_probs, batch_size);
            
            CUDA_CHECK_ERR(cudaDeviceSynchronize());
            sparce_categorical_cross_entropy_kernel<<<blocksPerGrid, threadsPerBlock>>>(dh_probs, d_target,
                                                                                 batch_size, num_classes,
                                                                                 epoch_loss);

            CUDA_CHECK_ERR(cudaDeviceSynchronize());

            backward(&network, d_target, d_input, dh_probs, batch_size);

            // Print detailed telemetry every 100 batches
            if (step % 100 == 0) {
                printf("\n--- Epoch %d | Batch %d ---\n", e, step);
                printf("  Current Batch Loss: %.6f\n", (*epoch_loss) / (step + 1));
                
                // Peek at the first and last layer gradients to check for Vanishing Gradients
                print_layer_stats("Output Layer Weights", network.layers[num_layers-1].grad_wrt_w, 
                                  network.layers[num_layers-1].input_size * 10);
                print_layer_stats("Input Layer Weights", network.layers[0].grad_wrt_w, 
                                  784 * network.layers[0].output_size);
            }

            update(&network, learning_rate, batch_size);

            step++;
        }
        
        // End of Epoch Summary
        printf("\n==============================\n");
        printf(" EPOCH %d COMPLETED\n", e);
        printf(" Average Loss: %.6f\n", *epoch_loss / (num_train_img / batch_size));
        printf("==============================\n\n");
    }

    // ======================
    // Free Memory
    // ======================

    for (int i = 0; i < network.num_layers; i++) {
        cudaFree(network.layers[i].activations);
        cudaFree(network.layers[i].preactivations);
        cudaFree(network.layers[i].biases);
        cudaFree(network.layers[i].weights);
        cudaFree(network.layers[i].grad_wrt_b);
        cudaFree(network.layers[i].grad_wrt_w);
        cudaFree(network.layers[i].grad_wrt_input);
    }

    cudaFree(network.layers);

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
