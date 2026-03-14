#include <time.h>

#include "kernels.cuh"
#include "mnist.c"

typedef struct {
    Layer *layers;
    int num_layers;
} Network;

void forward_test(Network *network, double *initial_inputs, int batch_size) {

    dim3 threadsPerBlock(16, 16);
    
    double *current_input = initial_inputs;
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
        cudaDeviceSynchronize();
    }
}

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

    // Initialize the batch size
    int batch_size = 64;

    printf("Initializing network\n");
    int layer_sizes[] = {784, 128, 64, 10};
    int num_layers = sizeof(layer_sizes) / sizeof(layer_sizes[0]);
    int num_classes = 10;
    Network network = create_network_layers(layer_sizes, num_layers);

    // Allocate probability array (array after softmax) for the host and device
    double *dh_probs;
    cudaMallocManaged((void**)&dh_probs, sizeof(double) * num_classes * batch_size);
    
    // allocate memory on the device (GPU) for inputs
    double *d_input;
    cudaMalloc((void**)&d_input, sizeof(double) * 784 * batch_size);

    // allocate memory on the device for target valeus
    double *d_target;
    cudaMalloc((void**)&d_target, sizeof(double) * batch_size);
    int epochs = 10;
    double learning_rate = 0.01;

    printf("Started training\n");
    for (int e = 0; e < epochs; e++) {

        double prediction_precent = 0;
        int correct_predictions = 0;
        double epoch_loss = 0.0;


        int N = 60000;
        for (int i = 0; i < N; i += batch_size) {
            // 1. COPY INPUT: Move current image data to GPU
            cudaMemcpy(d_input, train_image + (i * 784), sizeof(double) * 784 * batch_size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_target, train_image_label + i, sizeof(int) * batch_size, cudaMemcpyHostToDevice);

            // 2. FORWARD: Pass the GPU buffer to your forward function
            forward_test(&network, d_input, batch_size);

            int threadsPerBlock = 256;
            int blocksPerGrid = (batch_size + threadsPerBlock -1) / threadsPerBlock;
            // 3. SOFTMAX: Calculate the softmax in the GPU
            double *output_layer_activations_ptr = network.layers[network.num_layers -1].activations;
            softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(output_layer_activations_ptr, num_classes, dh_probs, batch_size);

            cudaDeviceSynchronize();
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
