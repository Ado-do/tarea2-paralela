#include "kernels.cuh"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdio.h>

// Tamaño de bloque para Tiling en matriz de covarianza
#define TILE_SIZE 32

using namespace std;

// Kernel para calcular el vector promedio de todas las imágenes
// *Cada hilo procesa un componente 'j' del vector
__global__ void compute_average_kernel(float* d_dataset, float* d_avg,
                                       int num_images, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n)
        return;

    float sum = 0.0f;
    for (int k = 0; k < num_images; k++) {
        sum += d_dataset[k * n + j];
    }
    d_avg[j] = sum / (float)num_images;
}

// Kernel para centrar los datos restando el vector promedio
// *Cada hilo procesa un elemento específico de la matriz d_dataset
__global__ void center_data_kernel(float* d_dataset, float* d_avg,
                                   int num_images, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_images * n;
    if (idx >= total_elements)
        return;

    int j = idx % n;
    d_dataset[idx] -= d_avg[j];
}

// Kernel para calcular la matriz de covarianza utilizando Tiling (memoria
// compartida) Matriz de Covarianza C = (V * V^T) / m V es de tamaño n x m (en
// memoria está guardado de forma que V[j, k] = d_dataset[k*n + j])
__global__ void compute_covariance_tiled_kernel(float* d_dataset, float* d_cov,
                                                int num_images, int n) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Coordenadas globales en la matriz de covarianza (n x n)
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    float sum = 0.0f;

    // Iteramos sobre los lotes de tamaño TILE_SIZE en la dimensión 'num_images'
    int num_tiles = (num_images + TILE_SIZE - 1) / TILE_SIZE;
    for (int m_step = 0; m_step < num_tiles; ++m_step) {
        __shared__ float s_A[TILE_SIZE][TILE_SIZE];
        __shared__ float s_B[TILE_SIZE][TILE_SIZE];

        // Cargar tile de A (V) a memoria compartida
        int k_A = m_step * TILE_SIZE + tx; // columna en A (índice de imagen k)
        if (row < n && k_A < num_images) {
            s_A[ty][tx] = d_dataset[k_A * n + row];
        } else {
            s_A[ty][tx] = 0.0f;
        }

        // Cargar tile de B (V^T) a memoria compartida
        int k_B = m_step * TILE_SIZE + ty; // fila en B (índice de imagen k)
        if (col < n && k_B < num_images) {
            s_B[ty][tx] = d_dataset[k_B * n + col];
        } else {
            s_B[ty][tx] = 0.0f;
        }

        __syncthreads(); // sincronizar antes de multiplicar

        // Multiplicar los tiles y acumular resultados localmente
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += s_A[ty][i] * s_B[i][tx];
        }

        __syncthreads(); // sincronizar antes de cargar el siguiente tile
    }

    // Escribir el resultado final en memoria global
    if (row < n && col < n) {
        d_cov[row * n + col] = sum / (float)num_images;
    }
}

// Función para verificar la correctitud en CPU
void verify_correctness_cpu(float* h_dataset_orig, float* h_cov_gpu,
                            int num_images, int n) {
    cout << "\n* Verificando correctitud en CPU\n";
    cout << "Calculando en CPU (esto puede tardar unos segundos)..." << endl;

    float* h_dataset_cpu = new float[num_images * n];
    float* h_avg_cpu = new float[n]();
    float* h_cov_cpu = new float[n * n]();

    // Copiar dataset original
    for (int i = 0; i < num_images * n; ++i) {
        h_dataset_cpu[i] = h_dataset_orig[i];
    }

    // Calcular promedio
    for (int k = 0; k < num_images; ++k) {
        for (int j = 0; j < n; ++j) {
            h_avg_cpu[j] += h_dataset_cpu[k * n + j];
        }
    }
    for (int j = 0; j < n; ++j) {
        h_avg_cpu[j] /= (float)num_images;
    }

    // Centrar datos
    for (int k = 0; k < num_images; ++k) {
        for (int j = 0; j < n; ++j) {
            h_dataset_cpu[k * n + j] -= h_avg_cpu[j];
        }
    }

    // Matriz de covarianza
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < num_images; ++k) {
                sum += h_dataset_cpu[k * n + row] * h_dataset_cpu[k * n + col];
            }
            h_cov_cpu[row * n + col] = sum / (float)num_images;
        }
    }

    // Comparar resultados
    float max_error = 0.0f;
    for (int i = 0; i < n * n; ++i) {
        float diff = abs(h_cov_cpu[i] - h_cov_gpu[i]);
        if (diff > max_error) {
            max_error = diff;
        }
    }

    cout << "Error máximo absoluto entre CPU y GPU: " << max_error << endl;
    cout << "Resultado: " << ((max_error < 1e-4) ? "CORRECTO" : "INCORRECTO")
         << endl;

    delete[] h_dataset_cpu;
    delete[] h_avg_cpu;
    delete[] h_cov_cpu;
}

// Implementación del Experimento 1 (CUDA Clásico Sincrónico)
void run_experiment_1(float* h_dataset, int num_images, int n,
                      bool check_correctness) {
    size_t dataset_bytes = num_images * n * sizeof(float);
    size_t avg_bytes = n * sizeof(float);
    size_t cov_bytes = (size_t)n * n * sizeof(float);

    float *d_dataset, *d_avg, *d_cov;
    float* h_cov; // matriz resultante en el host

    // Alojar memoria en el host para el resultado
    h_cov = (float*)malloc(cov_bytes);
    if (!h_cov) {
        cerr << "Error: No se pudo alojar la matriz de covarianza en el host "
                "(tamaño "
             << cov_bytes / (1024 * 1024) << " MB)" << endl;
        return;
    }

    cout << "Asignando "
         << (dataset_bytes + avg_bytes + cov_bytes) / (1024.0 * 1024.0)
         << " MB de VRAM..." << endl;

    // Alojar memoria en el device (GPU) usando cudaMalloc
    cudaMalloc((void**)&d_dataset, dataset_bytes);
    cudaMalloc((void**)&d_avg, avg_bytes);
    cudaMalloc((void**)&d_cov, cov_bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // * Copia H2D (Dataset) sincrónica
    cudaEventRecord(start);
    cudaMemcpy(d_dataset, h_dataset, dataset_bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_h2d = 0;
    cudaEventElapsedTime(&time_h2d, start, stop);
    //cout << "Tiempo de copia H2D (Dataset): " << time_h2d << " ms" << endl;

    // * Ejecución de Kernels
    cudaEventRecord(start);

    // 1 Kernel de Promedio
    int block_size = 256;
    int grid_size_avg = (n + block_size - 1) / block_size;
    compute_average_kernel<<<grid_size_avg, block_size>>>(d_dataset, d_avg,
                                                          num_images, n);

    // 2 Kernel de Centrado
    int total_elements = num_images * n;
    int grid_size_center = (total_elements + block_size - 1) / block_size;
    center_data_kernel<<<grid_size_center, block_size>>>(d_dataset, d_avg,
                                                         num_images, n);

    // 3 Kernel de Covarianza (Tiling en memoria compartida)
    dim3 block_cov(TILE_SIZE, TILE_SIZE);
    dim3 grid_cov((n + TILE_SIZE - 1) / TILE_SIZE,
                  (n + TILE_SIZE - 1) / TILE_SIZE);
    compute_covariance_tiled_kernel<<<grid_cov, block_cov>>>(d_dataset, d_cov,
                                                             num_images, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_kernels = 0;
    cudaEventElapsedTime(&time_kernels, start, stop);
    //cout << "Tiempo neto de cómputo (Kernels): " << time_kernels << " ms"
    //     << endl;

    // * Copia D2H (Covarianza) sincrónica
    cudaEventRecord(start);
    cudaMemcpy(h_cov, d_cov, cov_bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_d2h = 0;
    cudaEventElapsedTime(&time_d2h, start, stop);
    //cout << "Tiempo de copia D2H (Covarianza): " << time_d2h << " ms" << endl;

    cout << "[METRICAS]," << num_images << "," << time_h2d << ","
         << time_kernels << "," << time_d2h << endl;

    // * Verificación en CPU (Opcional)
    if (check_correctness) {
        verify_correctness_cpu(h_dataset, h_cov, num_images, n);
    }

    // Liberar recursos
    cudaFree(d_dataset);
    cudaFree(d_avg);
    cudaFree(d_cov);
    free(h_cov);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
