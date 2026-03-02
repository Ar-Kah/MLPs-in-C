/*
 First implementation of MNSIT data set for C
Takafumi Hoiruchi. 2018.
https://github.com/takafumihoriuchi/MNIST_for_C

Convert Takafumis verion to work with cuda
Aaro Karhu 2026
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

// set appropriate path for data
#define TRAIN_IMAGE "/home/omakone3/Programming/C/MLP/data/train-images-idx3-ubyte"
#define TRAIN_LABEL "/home/omakone3/Programming/C/MLP/data/train-labels-idx1-ubyte"
#define TEST_IMAGE  "/home/omakone3/Programming/C/MLP/data/t10k-images-idx3-ubyte"
#define TEST_LABEL  "/home/omakone3/Programming/C/MLP/data/t10k-labels-idx1-ubyte"

#define SIZE 784 // 28*28
#define NUM_TRAIN 60000
#define NUM_TEST 10000
#define LEN_INFO_IMAGE 4
#define LEN_INFO_LABEL 2

#define MAX_IMAGESIZE 1280
#define MAX_BRIGHTNESS 255
#define MAX_FILENAME 256
#define MAX_NUM_OF_IMAGES 1

// We use 1D pointers for everything. This allows us to use cudaMalloc/Memcpy directly.
unsigned char *train_image_char;
unsigned char *test_image_char;
unsigned char *train_label_char;
unsigned char *test_label_char;

double *train_image;
double *test_image;

// Initialize memory buffers
void init_mnist_buffers() {
    train_image_char = (unsigned char *)malloc(NUM_TRAIN * SIZE * sizeof(unsigned char));
    test_image_char = (unsigned char *)malloc(NUM_TEST * SIZE * sizeof(unsigned char));
    train_label_char = (unsigned char *)malloc(NUM_TRAIN * sizeof(unsigned char));
    test_label_char = (unsigned char *)malloc(NUM_TEST * sizeof(unsigned char));
    
    train_image = (double *)malloc(NUM_TRAIN * SIZE * sizeof(double));
    test_image = (double *)malloc(NUM_TEST * SIZE * sizeof(double));
}

// Flat reader function
void read_mnist_char(const char *file_path, int num_data, int arr_n, unsigned char *data_buffer) {
    int fd = open(file_path, O_RDONLY);
    if (fd == -1) { perror("open"); exit(1); }
    
    lseek(fd, (arr_n > 1) ? 16 : 8, SEEK_SET); 
    
    read(fd, data_buffer, num_data * arr_n * sizeof(unsigned char));
    close(fd);
}

// Convert 1D char array to 1D double array
void image_char2double(int num_data, unsigned char *data_char, double *data_double) {
    for (int i = 0; i < num_data * SIZE; i++) {
        data_double[i] = (double)data_char[i] / 255.0;
    }
}
