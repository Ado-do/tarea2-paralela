#include "../inc/dataloader.hpp"
#include <cstring>

DataLoader::DataLoader(int num_images, const std::string& dataset_dir, int target_width, int target_height)
    : dataset_dir(dataset_dir), num_images(num_images),
      target_width(target_width), target_height(target_height), is_loaded(false) {
    
    elements_per_image = target_width * target_height;
    total_elements = static_cast<size_t>(num_images) * elements_per_image;
    
    // Asignamos memoria contigua para todo el dataset
    cudaError_t err = cudaMallocHost((void**)&h_dataset, total_elements * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("Error en asignación de memoria: " + std::string(cudaGetErrorString(err)));
    }
}

DataLoader::~DataLoader() {
    if (h_dataset != nullptr) {
        cudaFreeHost(h_dataset);
        h_dataset = nullptr;
    }
}


void DataLoader::preprocess_data() {
    if (is_loaded) return;

    // Hardcodedo
    const int actual_files_on_disk = 100; 

    for (int k = 1; k <= num_images; k++) {
        size_t offset = static_cast<size_t>(k - 1) * elements_per_image;
        float* current_image_ptr = h_dataset + offset;

        if (k <= actual_files_on_disk) {
            // I/O y procesamiento CImg, esto corre 100 veces
            // *: DIV2K parte de 0801
            std::string filename = dataset_dir + "/0" + std::to_string(800 + k) + "x4.png";
            
            CImg<unsigned char> img(filename.c_str());
            CImg<unsigned char> gray_img = img.spectrum() > 1 ? img.get_RGBtoYCbCr().channel(0) : img;
            gray_img.resize(target_width, target_height, -100, -100, 5);

            cimg_forXY(gray_img, x, y) {
                current_image_ptr[y * target_width + x] = static_cast<float>(gray_img(x, y));
            }
        } else {
            // Duplicación en memoria via modulo (corre para todas las imagenes restantes sobre 100)
            // Calculamos el puntero fuente utilizando modulo para wrappear de vuelta a las primeras 100 imágenes
            size_t source_offset = static_cast<size_t>((k - 1) % actual_files_on_disk) * elements_per_image;
            float* source_ptr = h_dataset + source_offset;
            
            // memcpy!
            std::memcpy(current_image_ptr, source_ptr, elements_per_image * sizeof(float));
        }
    }
    is_loaded = true;
}

float* DataLoader::get_batch_pointer(int batch_index, int batch_size) const {
    size_t offset = static_cast<size_t>(batch_index) * batch_size * elements_per_image;
    size_t required = static_cast<size_t>(batch_size) * elements_per_image;
    if (offset + required > total_elements) {
        return nullptr;
    }
    return h_dataset + offset;
}

size_t DataLoader::get_batch_bytes(int batch_size) const {
    return static_cast<size_t>(batch_size) * elements_per_image * sizeof(float);
}

int DataLoader::get_elements_per_image() const {
    return elements_per_image;
}
