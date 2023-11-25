#ifndef CONVOLUTION_HPP
#define CONVOLUTION_HPP

/*
 * File Name: Convolution.hpp
 *
 * Author: Rabia OZKAN
 * GitHub: https://github.com/rabiaozkan
 * Creation Date: 2023-11-11
 * Last Update: 2023-11-25
 * License: Apache License, Version 2.0
 */

#include <opencv2/opencv.hpp>
#include <vector>

class Convolution {
public:
    std::vector<cv::Mat> applyConvolutionToBatch(const std::vector<cv::Mat>& batch, const std::vector<cv::Mat>& kernels);
    void multiKernelConvolve(const cv::Mat& input, const std::vector<cv::Mat>& kernels, std::vector<cv::Mat>& outputs);
};

#endif // CONVOLUTION_HPP
