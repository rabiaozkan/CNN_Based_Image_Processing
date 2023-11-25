#ifndef BATCH_NORMALIZATION_HPP
#define BATCH_NORMALIZATION_HPP

/*
 * File Name: BatchNormalization.hpp
 *
 * Author: Rabia OZKAN
 * GitHub: https://github.com/rabiaozkan
 * Creation Date: 2023-11-11
 * Last Update: 2023-11-25
 * License: Apache License, Version 2.0
 */

#include <opencv2/opencv.hpp>
#include <vector>

class BatchNormalization {
public:
    void applyBatchNorm(std::vector<cv::Mat>& batch, double epsilon, double scale, double shift);
};

#endif // BATCH_NORMALIZATION_HPP
