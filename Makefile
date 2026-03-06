##
# Project Title
#
# @file
# @version 0.1

main: main.cu
	nvcc main.cu -o mlp

run: mlp
	./mlp
# end
