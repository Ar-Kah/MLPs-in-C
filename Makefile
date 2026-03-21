# Compiler and flags
NVCC = nvcc
# -Iinclude tells the compiler to look in the /include folder for .cuh files
CFLAGS = -Iinclude 

# Directories
SRC_DIR = src
TEST_DIR = tests
BIN_DIR = bin

# Default target
all: $(BIN_DIR)/mlp

# Compile the main application
$(BIN_DIR)/mlp: $(SRC_DIR)/main.cu $(SRC_DIR)/kernels.cu $(SRC_DIR)/mnist.c
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(CFLAGS) $^ -o $@

# Run the main application
run: $(BIN_DIR)/mlp
	./$(BIN_DIR)/mlp

# Compile and run the tests
test: $(TEST_DIR)/test.cu $(SRC_DIR)/kernels.cu $(SRC_DIR)/mnist.c
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(CFLAGS)-w $^ -o $(BIN_DIR)/run_test
	./$(BIN_DIR)/run_test

# Clean up compiled files
clean:
	rm -rf $(BIN_DIR)

# Tell Make these aren't real files
.PHONY: all run test clean
