#include <iostream>
#include <exception>
#include <vector>
#include "../inc/dataloader.hpp"
#include "../inc/kernels.cuh"

using namespace std;

int main (int argc, char *argv[]) {
    cout << "* BENCHMARK Y PRUEBAS FINALES\n";

    // Default es 100
    int target_num_images = 100; 
    if (argc > 1) {
        target_num_images = std::stoi(argv[1]);
    }

    const string dataset_dir = "data/DIV2K_valid_LR_bicubic_X4";
    const int target_width = 128;
    const int target_height = 128;

    try {
        cout << "[INFO] Inicializando DataLoader global para el benchmark..." << endl;
        DataLoader loader(100, dataset_dir, target_width, target_height);
        loader.preprocess_data();
        
        int n = loader.get_elements_per_image();
        
        cout << "\n----------------------------------------\n";
        cout << "[INFO] Evaluando con batch de " << target_num_images << " imágenes..." << endl;

        float* batch_ptr = loader.get_batch_pointer(0, target_num_images);
        if (batch_ptr == nullptr) {
            cerr << "[ERROR] Puntero inválido para " << target_num_images << " imágenes." << endl;
            return 1;
        }

        // Experimento 1: CUDA clásico sincrónico
        cout << ">>> Ejecutando Experimento 1..." << endl;
        run_experiment_1(batch_ptr, target_num_images, n, false);

        // TODO: Agregar experimento 2
        // cout << ">>> Ejecutando Experimento 2..." << endl;
        // run_experiment_2(batch_ptr, target_num_images, n, false);

        cout << "\n========================================\n";
        cout << "[INFO] Benchmark completado exitosamente." << endl;

    } catch (const exception& e) {
        cerr << "[ERROR FATAL] " << e.what() << endl;
        return 1;
    }

    return 0;
}
