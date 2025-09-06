#pragma once
#include <cuda_runtime.h>

// Generic convolution (input: unsigned char interleaved RGB etc.; output same format)
void convolve_cuda(const unsigned char* input, unsigned char* output,
                   int width, int height, int channels,
                   const float* host_kernel, int ksize);

// Convenience wrappers for 3x3 filters
void sobel_magnitude_cuda(const unsigned char* input, unsigned char* output_gray,
                          int width, int height, int channels);

void prewitt_magnitude_cuda(const unsigned char* input, unsigned char* output_gray,
                            int width, int height, int channels);

void laplacian_cuda(const unsigned char* input, unsigned char* output_gray,
                    int width, int height, int channels);
