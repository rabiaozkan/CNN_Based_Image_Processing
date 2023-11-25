/*
 * File Name: Convolution.cpp
 * Purpose: Implements convolution operations for image processing.
 *          This file provides functionality to apply convolution using multiple kernels
 *          to a batch of images, suitable for image filtering and feature extraction in CNN workflows.
 *
 * Author: Rabia OZKAN
 * GitHub: https://github.com/rabiaozkan
 * Creation Date: 2023-11-11
 * Last Update: 2023-11-25
 * License: Apache License, Version 2.0
 */

#include "Convolution.hpp"

// Apply convolution to a batch of images
std::vector<cv::Mat> Convolution::applyConvolutionToBatch(const std::vector<cv::Mat>& batch, const std::vector<cv::Mat>& kernels) {
    std::vector<cv::Mat> batchConvResults;
    for (const auto& image : batch) {
        std::vector<cv::Mat> convResults;
        multiKernelConvolve(image, kernels, convResults);

        cv::Mat combinedResult;
        cv::merge(convResults, combinedResult);
        batchConvResults.push_back(combinedResult);
    }
    return batchConvResults;
}

// Function for convolution operation
void Convolution::multiKernelConvolve(const cv::Mat& input, const std::vector<cv::Mat>& kernels, std::vector<cv::Mat>& outputs) {
    if (input.empty()) {
        std::cerr << "Error: Input image is empty." << std::endl;
        return;
    }

    // Check for valid kernels
    for (const auto& kernel : kernels) {
        if (kernel.empty() or kernel.rows != kernel.cols) {
            std::cerr << "Error: Invalid kernel sizes." << std::endl;
            continue;
        }
        cv::Mat output;
        cv::filter2D(input, output, -1, kernel); // Apply the 2D filter
        outputs.push_back(output);
    }
}