#pragma once
#include <cuda_runtime.h>

// RGB <-> YCbCr
void rgb_to_ycbcr_cuda(const unsigned char* input, unsigned char* output,
                       int width, int height, int channels);

void ycbcr_to_rgb_cuda(const unsigned char* input, unsigned char* output,
                       int width, int height, int channels);

// RGB <-> HSV
void rgb_to_hsv_cuda(const unsigned char* input, unsigned char* output,
                     int width, int height, int channels);

void hsv_to_rgb_cuda(const unsigned char* input, unsigned char* output,
                     int width, int height, int channels);
