#include <iostream>
#include <exception>
#include <vector>
#include <string>
#include "../inc/dataloader.hpp"
#include "../inc/kernels.cuh"

using namespace std;

int main (int argc, char *argv[]) {
    cout << "* BENCHMARK Y PRUEBAS FINALES\n";

    // Valores por defecto
    int exp_id = 1; 
    int target_num_images = 100; 
    int batch_size = 25;
    int num_streams = 4;

    // Leer parámetros: ./benchmark <exp_id> <num_images> <batch_size> <num_streams>
    if (argc > 1) exp_id = std::stoi(argv[1]);
    if (argc > 2) target_num_images = std::stoi(argv[2]);
    if (argc > 3) batch_size = std::stoi(argv[3]);
    if (argc > 4) num_streams = std::stoi(argv[4]);

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

        // Switch de control de experimentos
        if (exp_id == 1) {
            cout << ">>> Ejecutando Experimento 1..." << endl;
            run_experiment_1(batch_ptr, target_num_images, n, false);
        } else if (exp_id == 2) {
            cout << "\n>>> Ejecutando Experimento 2 (Streams: " << num_streams 
            << ", Batch: " << batch_size << ")" << endl;
            run_experiment_2(batch_ptr, target_num_images, n, batch_size, num_streams, false);
        } else {
            cerr << "[ERROR] ID de experimento desconocido." << endl;
            return 1;
        }

        cout << "\n========================================\n";
        cout << "[INFO] Benchmark completado exitosamente." << endl;

    } catch (const exception& e) {
        cerr << "[ERROR FATAL] " << e.what() << endl;
        return 1;
    }

    return 0;
}