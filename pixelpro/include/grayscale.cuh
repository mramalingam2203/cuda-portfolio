#pragma once

#include <cuda_runtime.h>

void rgb_to_grayscale_cuda(const unsigned char* input, unsigned char* output,
                           int width, int height, int channels);