#include <iostream>
#include <exception>
#include "../inc/dataloader.hpp"
#include "../inc/kernels.cuh"

using namespace std;

int main (int argc, char *argv[]) {
    cout << "* TEST DE CORRECTITUD (MÓDULOS)\n";

    // Muestra reducida para verificar rápidamente que los resultados matemáticos sean correctos
    const int num_images = 16; 
    const string dataset_dir = "data/DIV2K_valid_LR_bicubic_X4";
    const int target_width = 128;
    const int target_height = 128;

    try {
        cout << "[INFO] Inicializando DataLoader ..." << endl;
        DataLoader loader(num_images, dataset_dir, target_width, target_height);
        
        cout << "[INFO] Preprocesando dataset..." << endl;
        loader.preprocess_data();

        int n = loader.get_elements_per_image();
        float* batch_ptr = loader.get_batch_pointer(0, num_images);
        
        if (batch_ptr == nullptr) {
            cerr << "[ERROR] Falló la resolución del puntero del batch." << endl;
            return 1;
        }
        
        cout << "[INFO] Elementos por imagen (n): " << n << endl;
        cout << "[INFO] Ejecutando Experimento 1 con verificación en CPU..." << endl;
        
        // Pasamos true para check_correctness
        run_experiment_1(batch_ptr, num_images, n, true);
        
        cout << "\n[INFO] Pruebas de correctitud finalizadas." << endl;

    } catch (const exception& e) {
        cerr << "[ERROR FATAL] " << e.what() << endl;
        return 1;
    }

    return 0;
}
