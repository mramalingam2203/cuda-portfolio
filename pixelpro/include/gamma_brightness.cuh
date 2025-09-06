#pragma once
#include <cuda_runtime.h>

// Brightness & Contrast scaling
void brightness_scale_cuda(const unsigned char* input, unsigned char* output,
                           int width, int height, int channels,
                           float alpha, float beta);

// Gamma Correction
void gamma_correction_cuda(const unsigned char* input, unsigned char* output,
                           int width, int height, int channels,
                           float gamma);
