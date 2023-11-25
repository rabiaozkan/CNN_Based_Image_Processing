/*
 * File Name: main.cpp
 * Purpose: To implement a CNN-based image processing workflow.
 *          This file loads images, performs preprocessing, applies convolution,
 *          batch normalization, and ReLU activation. It calculates and saves the results.
 *
 * Used Modules:
 *   - Timer.hpp: For time measurement
 *   - ImageProcessor.hpp: For image loading and preprocessing
 *   - Convolution.hpp: To apply convolution to images
 *   - BatchNormalization.hpp: For batch normalization operations
 *   - Activation.hpp: For activation functions
 *
 * Author: Rabia OZKAN
 * GitHub: https://github.com/rabiaozkan
 * Creation Date: 2023-11-11
 * Last Update: 2023-11-25
 * License: Apache License, Version 2.0
 */

#include "Timer.hpp"
#include "ImageProcessor.hpp"
#include "Convolution.hpp"
#include "BatchNormalization.hpp"
#include "Activation.hpp"
#include <iostream>

int main() {
    Timer timer;
    ImageProcessor imageProcessor;
    Convolution convolution;
    BatchNormalization batchNorm;
    Activation activation;

    std::string directoryPath = "D:/CNN_Based_Image_Processing/images";
    int width = 512;
    int height = 512;
    int imgCount = 0;

    double epsilon = 1e-7;
    double scale = 1.0;
    double shift = 0.0;

    // Start timer
    auto startTotal = timer.startTimer();

    // List and load images from directory
    auto start = timer.startTimer();
    std::vector<std::string> imagePaths = imageProcessor.listFilesInDirectory(directoryPath, imgCount);
    std::cout << "Directory listing time: " << timer.stopTimer(start) << " ms." << std::endl;

    // Process images
    start = timer.startTimer();
    std::vector<cv::Mat> batch = imageProcessor.preprocessImages(imagePaths, width, height, imgCount);
    std::cout << "Preprocessing time: " << timer.stopTimer(start) << " ms." << std::endl;

    // Kernel definitions
    cv::Mat kernel1 = (cv::Mat_<float>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
    cv::Mat kernel2 = (cv::Mat_<float>(3, 3) << 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11);
    std::vector<cv::Mat> combinedKernels = { kernel1, kernel2 };

    // Apply convolution to the entire batch
    start = timer.startTimer();
    auto batchConvResults = convolution.applyConvolutionToBatch(batch, combinedKernels);
    std::cout << "Convolution time: " << timer.stopTimer(start) << " ms." << std::endl;

    // Batch Normalization
    start = timer.startTimer();
    batchNorm.applyBatchNorm(batchConvResults, epsilon, scale, shift);
    std::cout << "Batch normalization time: " << timer.stopTimer(start) << " ms." << std::endl;

    // ReLU activation
    start = timer.startTimer();
    activation.relu(batchConvResults);
    std::cout << "ReLU activation time: " << timer.stopTimer(start) << " ms." << std::endl;

    // Total time
    std::cout << "Total time: " << timer.stopTimer(startTotal) << " ms." << std::endl;

    // Visualize and save images
    std::string outputDirectory = "D:/CNN_Based_Image_Processing/results_on_cpu";
    imageProcessor.visualizeAndSaveResults(batchConvResults, outputDirectory);

    return 0;
}