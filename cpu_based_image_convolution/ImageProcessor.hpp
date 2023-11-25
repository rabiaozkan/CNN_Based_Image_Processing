#ifndef IMAGE_PROCESSOR_HPP
#define IMAGE_PROCESSOR_HPP

/*
 * File Name: ImageProcessor.hpp
 *
 * Author: Rabia OZKAN
 * GitHub: https://github.com/rabiaozkan
 * Creation Date: 2023-11-11
 * Last Update: 2023-11-25
 * License: Apache License, Version 2.0
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

class ImageProcessor {
public:
    std::vector<cv::Mat> preprocessImages(const std::vector<std::string>& imagePaths, int targetWidth, int targetHeight, int batchSize);
    void visualizeAndSaveResults(const std::vector<cv::Mat>& batch, const std::string& outputDirectory);
    std::vector<std::string> listFilesInDirectory(const std::string& directoryPath, int& imgCount);
};

#endif // IMAGE_PROCESSOR_HPP