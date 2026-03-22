#ifndef MNIST_H_
#define MNIST_H_


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>

// set appropriate path for data
#define TRAIN_IMAGE "data/train-images-idx3-ubyte" 
#define TRAIN_LABEL "data/train-labels-idx1-ubyte"
#define TEST_IMAGE  "data/t10k-images-idx3-ubyte"
#define TEST_LABEL  "data/t10k-labels-idx1-ubyte"

#define SIZE 784 // 28*28
#define NUM_TRAIN 60000
#define NUM_TEST 10000
#define LEN_INFO_IMAGE 4
#define LEN_INFO_LABEL 2

#define MAX_IMAGESIZE 1280
#define MAX_BRIGHTNESS 255
#define MAX_FILENAME 256
#define MAX_NUM_OF_IMAGES 1

#ifdef __cplusplus
extern "C" {
#endif

// We use 1D pointers for everything. This allows us to use cudaMalloc/Memcpy directly.
extern unsigned char *train_image_char;
extern unsigned char *test_image_char;
extern unsigned char *train_label_char;
extern unsigned char *test_label_char;

extern float *train_image;
extern float *test_image;
extern int *train_image_label;
extern int *test_image_label;


void init_mnist_buffers();

void read_mnist_char(const char*, int, int, unsigned char*);

void image_char2float(int, unsigned char*, float*);

void label_char2int(int, unsigned char*, int*);

#ifdef __cplusplus
}
#endif

#endif // MNIST_H_
