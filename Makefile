# Config de Compilador 
NVCC = nvcc
CXX = g++

# Arquitectura GPU
NVCC_ARCH = 

# Flags e Includes
CUDA_PATH = /usr/local/cuda
INCLUDES = -Iinc -Ithird_party -I$(CUDA_PATH)/include
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra $(INCLUDES)
NVCCFLAGS = -O3 -std=c++17 $(NVCC_ARCH) -Xcompiler "-Wall -Wextra" $(INCLUDES)

# linkage 
LDFLAGS = -lX11 -lpthread -ljpeg -lpng

# Directorios
SRC_DIR = src
INC_DIR = inc
OBJ_DIR = obj
BIN_DIR = bin

# Ejecutables
TEST_TARGET = $(BIN_DIR)/tests
BENCH_TARGET = $(BIN_DIR)/benchmark

# Common sources (excluimos los main)
COMMON_CPP = $(SRC_DIR)/dataloader.cpp
COMMON_CU  = $(SRC_DIR)/kernels.cu

# Mains
TEST_MAIN = $(SRC_DIR)/tests.cpp
FINAL_MAIN = $(SRC_DIR)/benchmark.cpp

# Objects
COMMON_OBJ = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(COMMON_CPP)) \
             $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(COMMON_CU))
TEST_OBJ = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(TEST_MAIN))
FINAL_OBJ = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(FINAL_MAIN))

# Reglas de building
.PHONY: all prep clean tests benchmark

# Default target
all: prep $(TEST_TARGET) $(BENCH_TARGET)

# Asegurar que los directorios existen
prep:
	@mkdir -p $(OBJ_DIR) $(BIN_DIR)

tests: $(TEST_TARGET)
benchmark: $(BENCH_TARGET)

# Linkeamos ejecutable de tests
$(TEST_TARGET): $(COMMON_OBJ) $(TEST_OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

# Linkeamos ejecutable de pruebas finales (benchmark)
$(BENCH_TARGET): $(COMMON_OBJ) $(FINAL_OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

# Compilamos .cpp
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compilamos objetos cuda
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
