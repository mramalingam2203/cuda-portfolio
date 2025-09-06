#include "grayscale.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void rgb_to_gray_kernel(const unsigned char* input, unsigned char* output,
                                   int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (y * width + x) * channels;

    if (x < width && y < height) {
        unsigned char r = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char b = input[idx + 2];
        output[y * width + x] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

void rgb_to_grayscale_cuda(const unsigned char* input, unsigned char* output,
                           int width, int height, int channels) {
    unsigned char *d_input, *d_output;
    size_t img_size = width * height * channels * sizeof(unsigned char);
    size_t gray_size = width * height * sizeof(unsigned char);

    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_output, gray_size);

    cudaMemcpy(d_input, input, img_size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    rgb_to_gray_kernel<<<grid, block>>>(d_input, d_output, width, height, channels);

    cudaMemcpy(output, d_output, gray_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
