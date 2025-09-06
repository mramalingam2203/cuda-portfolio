#include "colorspace.cuh"
#include <device_launch_parameters.h>

__device__ __forceinline__ unsigned char clamp_val(float v) {
    return (v < 0.0f ? 0 : (v > 255.0f ? 255 : (unsigned char)v));
}

// ---------------- RGB → YCbCr ----------------
__global__ void rgb_to_ycbcr_kernel(const unsigned char* input, unsigned char* output,
                                    int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (y * width + x) * channels;

    if (x < width && y < height) {
        float r = input[idx];
        float g = input[idx + 1];
        float b = input[idx + 2];

        float Y  =  0.299f*r + 0.587f*g + 0.114f*b;
        float Cb = 128.0f - 0.168736f*r - 0.331264f*g + 0.5f*b;
        float Cr = 128.0f + 0.5f*r - 0.418688f*g - 0.081312f*b;

        output[idx]     = clamp_val(Y);
        output[idx + 1] = clamp_val(Cb);
        output[idx + 2] = clamp_val(Cr);
    }
}

void rgb_to_ycbcr_cuda(const unsigned char* input, unsigned char* output,
                       int width, int height, int channels) {
    unsigned char *d_in, *d_out;
    size_t size = width * height * channels;

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, input, size, cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid((width + 15)/16, (height + 15)/16);

    rgb_to_ycbcr_kernel<<<grid, block>>>(d_in, d_out, width, height, channels);

    cudaMemcpy(output, d_out, size, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}

// ---------------- RGB → HSV ----------------
__global__ void rgb_to_hsv_kernel(const unsigned char* input, unsigned char* output,
                                  int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (y * width + x) * channels;

    if (x < width && y < height) {
        float r = input[idx]     / 255.0f;
        float g = input[idx + 1] / 255.0f;
        float b = input[idx + 2] / 255.0f;

        float maxc = fmaxf(r, fmaxf(g, b));
        float minc = fminf(r, fminf(g, b));
        float delta = maxc - minc;

        float h = 0.0f, s = 0.0f, v = maxc;

        if (delta > 1e-6f) {
            s = delta / maxc;
            if (maxc == r)
                h = 60.0f * fmodf(((g - b) / delta), 6.0f);
            else if (maxc == g)
                h = 60.0f * (((b - r) / delta) + 2.0f);
            else
                h = 60.0f * (((r - g) / delta) + 4.0f);
            if (h < 0) h += 360.0f;
        }

        output[idx]     = (unsigned char)(h / 2); // compress [0,360] → [0,180]
        output[idx + 1] = (unsigned char)(s * 255);
        output[idx + 2] = (unsigned char)(v * 255);
    }
}

void rgb_to_hsv_cuda(const unsigned char* input, unsigned char* output,
                     int width, int height, int channels) {
    unsigned char *d_in, *d_out;
    size_t size = width * height * channels;

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, input, size, cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid((width + 15)/16, (height + 15)/16);

    rgb_to_hsv_kernel<<<grid, block>>>(d_in, d_out, width, height, channels);

    cudaMemcpy(output, d_out, size, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}
