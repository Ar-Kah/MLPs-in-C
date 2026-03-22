/*
 First implementation of MNSIT data set for C
Takafumi Hoiruchi. 2018.
https://github.com/takafumihoriuchi/MNIST_for_C

Convert Takafumis verion to work with cuda
Aaro Karhu 2026
*/
#include "mnist.h"

unsigned char *train_image_char;
unsigned char *test_image_char;
unsigned char *train_label_char;
unsigned char *test_label_char;

float *train_image;
float *test_image;
int *train_image_label;
int *test_image_label;

// Initialize memory buffers
void init_mnist_buffers() {
    train_image_char = (unsigned char *)malloc(NUM_TRAIN * SIZE * sizeof(unsigned char));
    test_image_char = (unsigned char *)malloc(NUM_TEST * SIZE * sizeof(unsigned char));
    train_label_char = (unsigned char *)malloc(NUM_TRAIN * sizeof(unsigned char));
    test_label_char = (unsigned char *)malloc(NUM_TEST * sizeof(unsigned char));
    
    train_image = (float *)malloc(NUM_TRAIN * SIZE * sizeof(float));
    test_image = (float *)malloc(NUM_TEST * SIZE * sizeof(float));
    train_image_label = (int*)malloc(NUM_TRAIN * SIZE * sizeof(int));
    test_image_label = (int*)malloc(NUM_TEST * SIZE* sizeof(int));
}

// Flat reader function
void read_mnist_char(const char *file_path, int num_data, int arr_n, unsigned char *data_buffer) {
    int fd = open(file_path, O_RDONLY);
    if (fd == -1) { perror("open"); exit(1); }
    
    lseek(fd, (arr_n > 1) ? 16 : 8, SEEK_SET); 
    
    read(fd, data_buffer, num_data * arr_n * sizeof(unsigned char));
    close(fd);
}

// Convert 1D char array to 1D float array
void image_char2float(int num_data, unsigned char *data_char, float *data_float) {
    for (int i = 0; i < num_data * SIZE; i++) {
        data_float[i] = (float)data_char[i] / 255.0;
    }
}

// Convert raw byte labels to an integer array (0-9)
void label_char2int(int num_data, unsigned char *data_char, int *label_int) {
    for (int i = 0; i < num_data; i++) {
        // Simple cast: binary byte -> integer
        label_int[i] = (int)data_char[i];
    }
}
