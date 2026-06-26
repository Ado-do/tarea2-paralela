#include <iostream>
#include <exception>
#include "../inc/dataloader.hpp"

int main (int argc, char *argv[]) {

    const int num_images = 100;
    const std::string dataset_dir = "data";
    const int target_width = 128;
    const int target_height = 128;

    std::cout << "[INFO] Inicializando DataLoader ..." << std::endl;

    try {
        DataLoader loader(num_images, dataset_dir, target_width, target_height);
        
        std::cout << "[INFO] Preprocesando dataset redimensión a " 
                  << target_width << "x" << target_height << ")..." << std::endl;
        
        loader.preprocess_data();

        const int batch_size = 16;
        std::cout << "[INFO] Dataset cargado exitosamente." << std::endl;
        std::cout << "[INFO] Elementos por imagen (n): " << loader.get_elements_per_image() << std::endl;
        std::cout << "[INFO] Bytes totales por batch (S=" << batch_size << "): " 
                  << loader.get_batch_bytes(batch_size) << " bytes" << std::endl;

        float* batch_ptr = loader.get_batch_pointer(0, batch_size);
        if (batch_ptr != nullptr) {
            std::cout << "[INFO] Puntero resuelto satisfactoriamente." << std::endl;
        } else {
            std::cerr << "[ERROR] Resolución de puntero fallada." << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "[ERROR FATAL] " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
