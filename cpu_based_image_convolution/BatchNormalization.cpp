/*
 * File Name: BatchNormalization.cpp
 * Purpose: Provides batch normalization functionality for image processing in neural networks.
 *          This file includes a method to normalize a batch of images by adjusting
 *          and scaling the activations.
 *
 * Author: Rabia OZKAN
 * GitHub: https://github.com/rabiaozkan
 * Creation Date: 2023-11-11
 * Last Update: 2023-11-25
 * License: Apache License, Version 2.0
 */

#include "BatchNormalization.hpp"

void BatchNormalization::applyBatchNorm(std::vector<cv::Mat>& batch, double epsilon, double scale, double shift) {
    if (batch.empty()) {
        std::cerr << "Error: Cannot normalize an empty batch." << std::endl;
        return;
    }

    cv::Scalar meanTotal, stddevTotal;
    // Calculate mean and standard deviation for the entire batch
    for (const auto& image : batch) {
        cv::Scalar mean, stddev;
        cv::meanStdDev(image, mean, stddev);
        meanTotal += mean;
        stddevTotal += stddev;
    }
    meanTotal /= static_cast<double>(batch.size());
    stddevTotal /= static_cast<double>(batch.size());

    // Apply normalization to the batch
    for (auto& image : batch) {
        // Normalize each image in the batch
        //image = (image - meanTotal[0]) / (stddevTotal[0] + epsilon);
        image = scale * ((image - meanTotal[0]) / (stddevTotal[0] + epsilon)) + shift;
    }
}