##
# Project Title
#
# @file
# @version 0.1

main: main.cu mnist.c kernels.cu
	nvcc main.cu mnist.c kernels.cu -o mlp

run: mlp
	./mlp

test: test.cu kernels.cu
	nvcc test.cu kernels.cu -o run_test


# end
