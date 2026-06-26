#pragma once

#include <string>
#include <stdexcept>
#include <iostream>
#include <cuda_runtime.h>

#define cimg_display 0
#define cimg_use_jpeg
#include "../third_party/CImg/CImg.h"

using namespace cimg_library;

class DataLoader {
    private:
        float* h_dataset;
        std::string dataset_dir;
        int num_images;
        int target_width;
        int target_height;
        int elements_per_image;
        size_t total_elements;
        bool is_loaded;

    public:
        DataLoader(int num_images, const std::string& dataset_dir, int target_width, int target_height);
        ~DataLoader();

        DataLoader(const DataLoader&) = delete;
        DataLoader& operator=(const DataLoader&) = delete;

        void preprocess_data();

        float* get_batch_pointer(int batch_index, int batch_size) const;
        size_t get_batch_bytes(int batch_size) const;
        int get_elements_per_image() const;
};
