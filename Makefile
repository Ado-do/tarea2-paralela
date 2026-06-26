# Config de Compilador 
NVCC = nvcc
CXX = g++

# Arquitectura GPU
NVCC_ARCH = 

CXXFLAGS = -O3 -std=c++17 -Wall -Wextra
NVCCFLAGS = -O3 -std=c++17 $(NVCC_ARCH) -Xcompiler "-Wall -Wextra"

# linkage 
LDFLAGS = -lX11 -lpthread

# Directorio
SRC_DIR = .
OBJ_DIR = obj
BIN_DIR = bin

# Ejecutable
TARGET = $(BIN_DIR)/main

# Automáticamente reconode todos los .cpp en SRC_DIR
CPP_SOURCES = $(wildcard $(SRC_DIR)/*.cpp)

# Los mappea a object files
CPP_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(CPP_SOURCES))
OBJECTS = $(CPP_OBJECTS) 

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

# Compilamos
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compilamos objetos cuda
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
