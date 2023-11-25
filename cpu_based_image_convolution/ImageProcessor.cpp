/*
 * File Name: ImageProcessor.cpp
 * Purpose: Provides image processing functionalities for CNN-based workflows.
 *          Includes methods for listing images, preprocessing, and visualizing & saving results.
 *
 * Author: Rabia OZKAN
 * GitHub: https://github.com/rabiaozkan
 * Creation Date: 2023-11-11
 * Last Update: 2023-11-25
 * License: Apache License, Version 2.0
 */

#include "ImageProcessor.hpp"
#include <sstream>
#include <iomanip>

// List images in a directory
std::vector<std::string> ImageProcessor::listFilesInDirectory(const std::string& directoryPath, int& imgCount) {
    std::vector<std::string> files;

    // Check if directory exists and is accessible
    if (!std::filesystem::exists(directoryPath) || !std::filesystem::is_directory(directoryPath)) {
        std::cerr << "Error: Directory '" << directoryPath << "' not found or is not a directory." << std::endl;
        return files;
    }

    // Iterate through the directory and list .jpg files
    for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
        if (entry.path().extension() == ".jpg") {
            files.push_back(entry.path().string());
            imgCount++;
        }
    }
    return files;
}

// Preprocess images for input to a neural network or other processing
std::vector<cv::Mat> ImageProcessor::preprocessImages(const std::vector<std::string>& imagePaths, int targetWidth, int targetHeight, int batchSize) {
    std::vector<cv::Mat> batch;
    for (size_t i = 0; i < std::min(imagePaths.size(), static_cast<size_t>(batchSize)); ++i) {
        cv::Mat image = cv::imread(imagePaths[i], cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "Error: File " << imagePaths[i] << " could not be loaded or is empty." << std::endl;
            continue;
        }
        cv::resize(image, image, cv::Size(targetWidth, targetHeight));
        batch.push_back(image);
    }
    return batch;
}

// Function to visualize and save the results of image processing
void ImageProcessor::visualizeAndSaveResults(const std::vector<cv::Mat>& batch, const std::string& outputDirectory) {
    int fileCounter = 0;

    for (const auto& image : batch) {
        std::vector<cv::Mat> channels;
        cv::split(image, channels);

        for (size_t i = 0; i < channels.size(); i++) {
            cv::Mat displayImage;
            // Normalize the image and convert to CV_8U type
            cv::normalize(channels[i], displayImage, 0, 255, cv::NORM_MINMAX);
            displayImage.convertTo(displayImage, CV_8U);

            // Save the image
            std::stringstream ss;
            ss << outputDirectory << "/result_" << std::setfill('0') << std::setw(4) << fileCounter++ << "_channel_" << i << ".jpg";
            cv::imwrite(ss.str(), displayImage);
        }
    }
}