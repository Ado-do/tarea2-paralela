# Config de Compilador 
NVCC = nvcc
CXX = g++

# Arquitectura GPU
NVCC_ARCH = 

# Flags e Includes
# Añadimos -Iinc y -Ithird_party para que encuentre los .hpp y dependencias
CUDA_PATH = /usr/local/cuda
INCLUDES = -Iinc -Ithird_party -I$(CUDA_PATH)/include
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra $(INCLUDES)
NVCCFLAGS = -O3 -std=c++17 $(NVCC_ARCH) -Xcompiler "-Wall -Wextra" $(INCLUDES)

# linkage 
LDFLAGS = -lX11 -lpthread -ljpeg

# Directorios
SRC_DIR = src
INC_DIR = inc
OBJ_DIR = obj
BIN_DIR = bin

# Ejecutable
TARGET = $(BIN_DIR)/main

# Automáticamente reconoce todos los .cpp y .cu en SRC_DIR
CPP_SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
CU_SOURCES = $(wildcard $(SRC_DIR)/*.cu)

# Los mappea a object files
CPP_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(CPP_SOURCES))
CU_OBJECTS = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(CU_SOURCES))
OBJECTS = $(CPP_OBJECTS) $(CU_OBJECTS) 

# Reglas de building
.PHONY: all prep clean run

# Default target
all: prep $(TARGET)

# Ensure build directories exist
prep:
	@mkdir -p $(OBJ_DIR) $(BIN_DIR)

# Linkeamos objetos
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

# Compilamos .cpp
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compilamos objetos cuda
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
