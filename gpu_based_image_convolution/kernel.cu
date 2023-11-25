/*
 * File Name: kernel.cu
 * Purpose: To implement a CNN-based image processing workflow on GPU.
 *          This file includes functions for preprocessing images, performing convolution,
 *          batch normalization, and ReLU activation, and then visualizing and saving the results.
 *          It leverages CUDA for parallel processing on NVIDIA GPUs.
 *
 * Used Modules:
 *   - OpenCV for image processing
 *   - CUDA for parallel processing on GPU
 *
 * Author: Rabia OZKAN
 * GitHub: https://github.com/rabiaozkan
 * Creation Date: 2023-11-20
 * Last Update: 2023-11-25
 * License: Apache License, Version 2.0
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <Windows.h>

#include "opencv2/opencv.hpp"
#include <vector>
#include <string>

using namespace std;

// Kernel function definitions
__global__ void convolveKernel(float* d_img, float* d_kernel, float* d_dest, int width, int height, int kernelSize);
__global__ void batchnormKernel(float* d_Y, float* d_Z, int size, float scale, float shift, float epsilon, float mean, float variance);
__global__ void reluKernel(float* d_Z, float* d_V, int size);

// Other function definitions
float mean(const float* data, int size);
float variance(const float* data, int size, float meanValue);
void convolve(cv::Mat& img, cv::Mat& kernel, cv::Mat& dest);
void batchnorm(cv::Mat& Y, cv::Mat& Z, float scale, float shift, float epsilon);
void relu(cv::Mat& Z, cv::Mat& V);
void visualize(cv::Mat& V);
std::vector<cv::Mat> preprocess(std::vector<std::string> imagePaths);

const int threadsPerBlock = 256;

// Task 1: Preprocessing Step
std::vector<cv::Mat> preprocess(std::vector<std::string> imagePaths) {
	std::vector<cv::Mat> images;

	// Load each image from the path and resize it
	for (const auto& path : imagePaths) {
		cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
		if (img.empty()) {
			cerr << "Image could not be loaded: " << path << endl;
			continue;
		}
		cv::resize(img, img, cv::Size(512, 512));
		images.push_back(img);
	}

	return images;
}

// Batch Normalization Kernel
__global__ void batchnormKernel(float* d_Y, float* d_Z, int size, float scale, float shift, float epsilon, float mean, float variance) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		d_Z[idx] = scale * (d_Y[idx] - mean) / sqrtf(variance + epsilon) + shift;
	}
}

// Implementation of other functions
float mean(const float* data, int size) {
	float sum = 0;
	for (int i = 0; i < size; ++i) {
		sum += data[i];
	}
	return sum / size;
}

float variance(const float* data, int size, float meanValue) {
	float sum = 0;
	for (int i = 0; i < size; ++i) {
		sum += (data[i] - meanValue) * (data[i] - meanValue);
	}
	return sum / size;
}

// Helper function to check CUDA errors
void checkCudaErrors(cudaError_t err, const char* msg) {
	if (err != cudaSuccess) {
		cerr << "CUDA Error: " << msg << ", " << cudaGetErrorString(err) << endl;
		exit(EXIT_FAILURE);
	}
}

// Update Activation function to work in two dimensions
__global__ void reluKernel(float* d_Z, float* d_V, int width, int height) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = row * width + col;

	if (row < height && col < width) {
		d_V[idx] = max(0.0f, d_Z[idx]);
	}
}

// CUDA Kernel Functions
__global__ void convolveKernel(float* d_img, float* d_kernel, float* d_dest, int width, int height, int kernelSize) {
	// Calculate the column and row indices of the pixel to be processed by the thread
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	// Perform the operation if within the bounds of the image
	if (row < height && col < width) {
		float sum = 0.0f;
		int start = -kernelSize / 2;
		// Iterate over the kernel and apply the convolution operation
		for (int i = 0; i < kernelSize; ++i) {
			for (int j = 0; j < kernelSize; ++j) {
				int r = min(max(row + start + i, 0), height - 1);
				int c = min(max(col + start + j, 0), width - 1);
				sum += d_kernel[i * kernelSize + j] * d_img[r * width + c];
			}
		}
		d_dest[row * width + col] = sum;
	}
}

// Convolution operation
void convolve(cv::Mat& img, const std::vector<cv::Mat>& kernels, std::vector<cv::Mat>& dests) {
	// Create a separate output matrix for each kernel
	dests.clear();
	for (size_t i = 0; i < kernels.size(); ++i) {
		dests.push_back(cv::Mat(img.rows, img.cols, CV_32F, cv::Scalar(0)));
	}

	// Check the type of the input image
	if (img.type() != CV_32F) {
		img.convertTo(img, CV_32F);
	}

	const int imgSize = img.rows * img.cols * sizeof(float);

	// Allocate memory on the GPU
	float* d_img;
	checkCudaErrors(cudaMalloc(&d_img, imgSize), "Allocating d_img");
	checkCudaErrors(cudaMemcpy(d_img, img.ptr<float>(), imgSize, cudaMemcpyHostToDevice), "Copying img to d_img");

	for (size_t k = 0; k < kernels.size(); ++k) {
		const cv::Mat& kernel = kernels[k];
		cv::Mat& dest = dests[k];

		const int kernelSize = kernel.rows * kernel.cols * sizeof(float);
		float* d_kernel, * d_dest;

		checkCudaErrors(cudaMalloc(&d_kernel, kernelSize), "Allocating d_kernel");
		checkCudaErrors(cudaMalloc(&d_dest, imgSize), "Allocating d_dest");

		checkCudaErrors(cudaMemcpy(d_kernel, kernel.ptr<float>(), kernelSize, cudaMemcpyHostToDevice), "Copying kernel to d_kernel");

		dim3 threadsPerBlock(16, 16);
		dim3 numBlocks(static_cast<unsigned int>(ceil(img.cols / 16.0)), static_cast<unsigned int>(ceil(img.rows / 16.0)));
		convolveKernel << <numBlocks, threadsPerBlock >> > (d_img, d_kernel, d_dest, img.cols, img.rows, kernel.rows);

		checkCudaErrors(cudaMemcpy(dest.ptr<float>(), d_dest, imgSize, cudaMemcpyDeviceToHost), "Copying d_dest to dest");

		cudaFree(d_kernel);
		cudaFree(d_dest);
	}

	cudaFree(d_img);
}

// Batch normalization operation
void batchnorm(cv::Mat& Y, cv::Mat& Z, float scale, float shift, float epsilon) {
	if (Y.empty()) {
		cerr << "Empty input matrix for batch normalization." << endl;
		return;
	}

	// Compute the mean and variance
	float meanValue = mean(reinterpret_cast<float*>(Y.data), Y.rows * Y.cols);
	float varValue = variance(reinterpret_cast<float*>(Y.data), Y.rows * Y.cols, meanValue);

	// Allocate memory on the GPU
	float* d_Y, * d_Z;
	const int size = Y.rows * Y.cols;

	cudaMalloc((void**)&d_Y, size * sizeof(float));
	cudaMalloc((void**)&d_Z, size * sizeof(float));
	cudaMemcpy(d_Y, reinterpret_cast<float*>(Y.data), size * sizeof(float), cudaMemcpyHostToDevice);

	// Launch the batch normalization kernel
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	batchnormKernel << <blocksPerGrid, threadsPerBlock >> > (d_Y, d_Z, size, scale, shift, epsilon, meanValue, varValue);

	// Check for kernel launch errors
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		cerr << "batchnormKernel launch failed: " << cudaGetErrorString(cudaStatus) << endl;
		cudaFree(d_Y);
		cudaFree(d_Z);
		return;
	}

	// Retrieve the results
	Z.create(Y.rows, Y.cols, CV_32F);
	cudaMemcpy(Z.ptr<float>(), d_Z, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_Y);
	cudaFree(d_Z);
}

// ReLU activation function
void relu(cv::Mat& Z, cv::Mat& V) {
	if (Z.empty()) {
		cerr << "ReLU activation function empty input." << endl;
		return;
	}

	const int size = Z.rows * Z.cols;
	float* d_Z, * d_V;

	// Allocate memory on the GPU
	cudaMalloc((void**)&d_Z, size * sizeof(float));
	cudaMalloc((void**)&d_V, size * sizeof(float));

	// Copy data to the GPU
	cudaMemcpy(d_Z, Z.ptr<float>(), size * sizeof(float), cudaMemcpyHostToDevice);

	// Launch the ReLU kernel in a two-dimensional configuration
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks((Z.cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(Z.rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
	reluKernel << <numBlocks, threadsPerBlock >> > (d_Z, d_V, Z.cols, Z.rows);

	// Retrieve the results
	V.create(Z.rows, Z.cols, CV_32F); // Create the V matrix
	cudaMemcpy(V.ptr<float>(), d_V, size * sizeof(float), cudaMemcpyDeviceToHost);

	// Free GPU memory
	cudaFree(d_Z);
	cudaFree(d_V);
}

// Function to visualize and save the results
void visualizeAndSaveResults(const std::vector<cv::Mat>& batch, const std::string& outputDirectory) {
	int fileCounter = 0;

	for (const auto& image : batch) {
		std::vector<cv::Mat> channels;
		cv::split(image, channels);

		for (size_t i = 0; i < channels.size(); i++) {
			// Convert the image to CV_8U type and scale to the 0-255 range
			cv::Mat displayImage;
			if (channels[i].type() == CV_32F) {
				channels[i].convertTo(displayImage, CV_8U, 255.0);
			}
			else {
				displayImage = channels[i];
			}

			// Save the image
			std::stringstream ss;
			ss << outputDirectory << "/result_" << std::setfill('0') << std::setw(4) << fileCounter++ << "_channel_" << i << ".jpg";
			cv::imwrite(ss.str(), displayImage);
		}
	}
}

// Main function
int main() {
	// Timer events for performance measurement
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Image preprocessing
	cudaEventRecord(start);
	std::vector<std::string> imagePaths = { "D:/CNN_Based_Image_Processing/images/image1.jpg",
											"D:/CNN_Based_Image_Processing/images/image2.jpg",
											"D:/CNN_Based_Image_Processing/images/image3.jpg",
											"D:/CNN_Based_Image_Processing/images/image4.jpg" };

	std::vector<cv::Mat> preprocessedImages = preprocess(imagePaths);
	std::vector<cv::Mat> Results;

	// Define kernels
	float sobelData[9] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };
	float avgData[9] = { 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11 };
	cv::Mat sobelKernel(3, 3, CV_32F, sobelData);
	cv::Mat avgKernel(3, 3, CV_32F, avgData);
	std::vector<cv::Mat> kernels = { sobelKernel, avgKernel };

	// Example values
	float scale = 1.0f;
	float shift = 0.0f;
	float epsilon = 1e-7;

	// Process each image
	for (cv::Mat& img : preprocessedImages) {
		if (img.empty()) {
			cerr << "Empty image, cannot proceed." << endl;
			continue;
		}

		std::vector<cv::Mat> convResults;
		convolve(img, kernels, convResults);

		for (cv::Mat& dest : convResults) {
			cv::Mat Y, Z;

			// Batch Normalization
			batchnorm(dest, Y, scale, shift, epsilon);
			if (Y.empty()) {
				cerr << "Empty matrix after batch normalization." << endl;
				continue;
			}

			// Activation (ReLU)
			relu(Y, Z);
			if (Z.empty()) {
				cerr << "Empty matrix after ReLU activation." << endl;
				continue;
			}

			// Add result to Results vector for visualization
			Results.push_back(Z);
		}
	}

	// Stop the timer and calculate elapsed time
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float timerCNN = 0;
	cudaEventElapsedTime(&timerCNN, start, stop);
	cout << "Total processing time: " << timerCNN << " ms." << endl;

	// Release the timers
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Save and visualize the processed images
	std::string outputDirectory = "D:/CNN_Based_Image_Processing/results_on_gpu";
	visualizeAndSaveResults(Results, outputDirectory);

	return 0;
}