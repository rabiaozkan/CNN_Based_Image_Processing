# CNN_Based_Image_Processing
CNN Based Image Processing
Overview
This project showcases a Convolutional Neural Network (CNN)-based image processing workflow, implemented for both CPU and GPU environments. It includes key image processing operations such as preprocessing, convolution, batch normalization, and ReLU activation, highlighting the efficiency and versatility of CNNs in handling image data.

Structure
The project is structured into two primary components:

CPU Implementation
Directory: cpu_based_image_convolution
Main File: main.cpp
Description: Manages image preprocessing, convolution, batch normalization, and ReLU activation using CPU-based processing techniques.
GPU Implementation
Directory: gpu_based_image_convolution
Main File: kernel.cu
Description: Mirrors the functionality of the CPU version but employs CUDA for accelerated parallel processing on NVIDIA GPUs.
Dependencies
OpenCV: Utilized for image loading, preprocessing, and transformations.
CUDA Toolkit: Required for the GPU implementation, enabling parallel processing capabilities.
Getting Started
Clone the Repository
bash
Copy code
git clone https://github.com/rabiaozkan/CNN_Based_Image_Processing.git
Navigate
Choose the appropriate directory (cpu_based_image_convolution or gpu_based_image_convolution) based on the desired implementation.

Compilation
CPU Version:
bash
Copy code
g++ -std=c++11 main.cpp -o main_cpu `pkg-config --cflags --libs opencv4`
GPU Version:
bash
Copy code
nvcc kernel.cu -o kernel `pkg-config --cflags --libs opencv4`
Execution
CPU: ./main_cpu
GPU: ./kernel
Usage
Modify the image paths in the source files to use your own dataset. The project processes a batch of images through various CNN stages, producing processed output for each step.

Additional Resources
Medium Article on OpenCV and GPU Support: OpenCV4.5.4 GPU Support (C++) by Batuhan Hangün.
YouTube Video Series on CUDA Programming: CUDA Programming Tutorial Series by Nicolai Nielsen.
These resources provide additional context and depth, especially for users interested in the technical details of GPU processing and CUDA integration with image processing tasks.

Contributing
We encourage contributions! If you have suggestions or improvements, please fork the repository and submit a pull request with your updates.

License
Licensed under the Apache License, Version 2.0. For more details, see the LICENSE file.

Author
Rabia OZKAN - GitHub Profile
