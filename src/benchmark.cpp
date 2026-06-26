#include <iostream>
#include <exception>
#include <vector>
#include "../inc/dataloader.hpp"
#include "../inc/kernels.cuh"

using namespace std;

int main (int argc, char *argv[]) {
    cout << "* BENCHMARK Y PRUEBAS FINALES\n";

    // Lista de diferentes tamaños de batch a testear
    vector<int> test_sizes = {16, 32, 64, 100};
    const string dataset_dir = "data/DIV2K_valid_LR_bicubic_X4"; 
    const int target_width = 128;
    const int target_height = 128;

    try {
        cout << "[INFO] Inicializando DataLoader global para el benchmark..." << endl;
        // La máxima cantidad que evaluaremos es 100 en este caso base
        int max_images = 100;
        DataLoader loader(max_images, dataset_dir, target_width, target_height);
        loader.preprocess_data();
        
        int n = loader.get_elements_per_image();
        
        cout << "[INFO] Dataset listo. Elementos por imagen (n): " << n << endl;

        for (int num_images : test_sizes) {
            cout << "\n----------------------------------------\n";
            cout << "[INFO] Evaluando con batch de " << num_images << " imágenes..." << endl;

            float* batch_ptr = loader.get_batch_pointer(0, num_images);
            if (batch_ptr == nullptr) {
                cerr << "[ERROR] Puntero inválido para " << num_images << " imágenes." << endl;
                continue;
            }

            // Experimento 1: CUDA clásico sincrónico
            cout << ">>> Ejecutando Experimento 1..." << endl;
            // No verificamos correctitud en benchmark para medir el tiempo de forma pura
            run_experiment_1(batch_ptr, num_images, n, false);

            // TODO: Agregar experimento 2
            // cout << ">>> Ejecutando Experimento 2..." << endl;
            // run_experiment_2(batch_ptr, num_images, n, false);
        }

        cout << "\n========================================\n";
        cout << "[INFO] Benchmark completado exitosamente." << endl;

    } catch (const exception& e) {
        cerr << "[ERROR FATAL] " << e.what() << endl;
        return 1;
    }

    return 0;
}
