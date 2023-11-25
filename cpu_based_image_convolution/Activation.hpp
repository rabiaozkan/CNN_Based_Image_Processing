#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

/*
 * File Name: Activation.hpp
 *
 * Author: Rabia OZKAN
 * GitHub: https://github.com/rabiaozkan
 * Creation Date: 2023-11-11
 * Last Update: 2023-11-25
 * License: Apache License, Version 2.0
 */

#include <opencv2/opencv.hpp>
#include <vector>

class Activation {
public:
    void relu(std::vector<cv::Mat>& batch);
};

#endif // ACTIVATION_HPP
