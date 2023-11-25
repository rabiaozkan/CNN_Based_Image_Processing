/*
 * File Name: Activation.cpp
 * Purpose: Implements activation functions for image processing in neural networks.
 *          This file currently provides the ReLU (Rectified Linear Unit) activation function.
 *
 * Author: Rabia OZKAN
 * GitHub: https://github.com/rabiaozkan
 * Creation Date: 2023-11-11
 * Last Update: 2023-11-25
 * License: Apache License, Version 2.0
 */

#include "Activation.hpp"

// Apply ReLU activation to a batch of images
void Activation::relu(std::vector<cv::Mat>& batch) {
    if (batch.empty()) {
        std::cerr << "Error: Cannot perform activation on an empty batch." << std::endl;
        return;
    }

    // Apply ReLU (Rectified Linear Unit) to each image in the batch
    for (auto& image : batch) {
        cv::max(image, 0, image); // ReLU function: f(x) = max(0, x)
    }
}