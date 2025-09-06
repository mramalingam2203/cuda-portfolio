#include "gamma_brightness.cuh"
#include <device_launch_parameters.h>

__device__ __forceinline__ unsigned char clamp_val(float v) {
    return (v < 0.0f ? 0 : (v > 255.0f ? 255 : (unsigned char)v));
}

// ---------------- Brightness Scaling ----------------
__global__ void brightness_scale_kernel(const unsigned char* input,
                                        unsigned char* output,
                                        int width, int height, int channels,
                                        float alpha, float beta) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (y * width + x) * channels;

    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            float val = alpha * input[idx + c] + beta;
            output[idx + c] = clamp_val(val);
        }
    }
}

void brightness_scale_cuda(const unsigned char* input, unsigned char* output,
                           int width, int height, int channels,
                           float alpha, float beta) {
    unsigned char *d_in, *d_out;
    size_t size = width * height * channels;

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, input, size, cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid((width + 15)/16, (height + 15)/16);

    brightness_scale_kernel<<<grid, block>>>(d_in, d_out, width, height, channels, alpha, beta);

    cudaMemcpy(output, d_out, size, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}

// ---------------- Gamma Correction ----------------
__global__ void gamma_correction_kernel(const unsigned char* input,
                                        unsigned char* output,
                                        int width, int height, int channels,
                                        float gamma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (y * width + x) * channels;

    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            float norm = input[idx + c] / 255.0f;
            float corrected = powf(norm, gamma);
            output[idx + c] = clamp_val(corrected * 255.0f);
        }
    }
}

void gamma_correction_cuda(const unsigned char* input, unsigned char* output,
                           int width, int height, int channels,
                           float gamma) {
    unsigned char *d_in, *d_out;
    size_t size = width * height * channels;

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, input, size, cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid((width + 15)/16, (height + 15)/16);

    gamma_correction_kernel<<<grid, block>>>(d_in, d_out, width, height, channels, gamma);

    cudaMemcpy(output, d_out, size, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}
