#include "../inc/dataloader.hpp"

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

    for (int k = 1; k <= num_images; k++) {
        std::string filename = dataset_dir + "/0" + std::to_string(800 + k) + "x4.png";
        
        CImg<unsigned char> img(filename.c_str());
        
        // Convertimos a greyscale
        CImg<unsigned char> gray_img = img.spectrum() > 1 ? img.get_RGBtoYCbCr().channel(0) : img;

        // Downscaling a target_width y height
        gray_img.resize(target_width, target_height, -100, -100, 5); // 5 es bicubic interpolation

        // Populamos el dataset
        size_t offset = static_cast<size_t>(k) * elements_per_image;
        float* current_image_ptr = h_dataset + offset;

        cimg_forXY(gray_img, x, y) {
            current_image_ptr[y * target_width + x] = static_cast<float>(gray_img(x, y));
        }
    }
    is_loaded = true;
}

float* DataLoader::get_batch_pointer(int batch_index, int batch_size) const {
    size_t offset = static_cast<size_t>(batch_index) * batch_size * elements_per_image;
    if (offset >= total_elements) {
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
