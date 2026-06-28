#ifndef KERNELS_CUH
#define KERNELS_CUH

// Ejecuta el experimento 1
void run_experiment_1(float* h_dataset, int num_images, int n, bool check_correctness);

// Ejecuta el experimento 2 (Streams y Batches)
void run_experiment_2(float* h_dataset, int num_images, int n, int batch_size, int num_streams, bool check_correctness);

#endif