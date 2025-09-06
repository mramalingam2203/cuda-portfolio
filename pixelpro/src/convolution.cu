#include "convolution.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define CHECK_CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA Err %s:%d : %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while(0)

// Shared-memory tiled generic convolution kernel
// Assumptions:
// - channels: number of channels interleaved (1,3,4)
// - kernel is square (ksize x ksize), copied to device beforehand
// - input and output are packed row-major
extern "C" __global__
void conv_shared_kernel(const unsigned char* d_in, unsigned char* d_out,
                        int width, int height, int channels,
                        const float* __restrict__ d_kernel, int ksize)
{
    // tile dims
    const int TILE_X = 16;
    const int TILE_Y = 16;

    int half = ksize / 2;

    // coords of pixel this thread will compute
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * TILE_X;
    int by = blockIdx.y * TILE_Y;

    int x = bx + tx;
    int y = by + ty;

    // allocate shared memory for one channel tile with apron.
    // shared tile width = TILE_X + 2*half, height = TILE_Y + 2*half
    extern __shared__ unsigned char shmem[]; // will store for all channels sequentially
    // We'll compute per-channel by indexing into shmem block
    int tileW = TILE_X + 2*half;
    int tileH = TILE_Y + 2*half;
    int tileSize = tileW * tileH; // per-channel

    // load shared memory for each channel
    for (int c = 0; c < channels; ++c) {
        unsigned char* sh = shmem + c * tileSize;

        // Each thread loads multiple pixels into shared memory (cooperative)
        // map local coordinates in shared tile for this thread
        for (int oy = ty; oy < tileH; oy += blockDim.y) {
            for (int ox = tx; ox < tileW; ox += blockDim.x) {
                int img_x = bx + (ox - half);
                int img_y = by + (oy - half);

                // clamp coordinates to image border
                img_x = img_x < 0 ? 0 : (img_x >= width ? width-1 : img_x);
                img_y = img_y < 0 ? 0 : (img_y >= height ? height-1 : img_y);

                int img_idx = (img_y * width + img_x) * channels + c;
                sh[oy * tileW + ox] = d_in[img_idx];
            }
        }
    }

    __syncthreads();

    // compute convolution for pixel if inside image
    if (x < width && y < height) {
        for (int c = 0; c < channels; ++c) {
            unsigned char* sh = shmem + c * tileSize;
            float accum = 0.0f;
            for (int ky = 0; ky < ksize; ++ky) {
                int sy = ty + ky;
                for (int kx = 0; kx < ksize; ++kx) {
                    int sx = tx + kx;
                    float kval = d_kernel[ky * ksize + kx];
                    unsigned char px = sh[sy * tileW + sx];
                    accum += kval * (float)px;
                }
            }
            // clamp and write
            int out_idx = (y * width + x) * channels + c;
            int v = (int)(accum + 0.5f);
            if (v < 0) v = 0;
            if (v > 255) v = 255;
            d_out[out_idx] = (unsigned char)v;
        }
    }
}

// Host wrapper for generic convolution
void convolve_cuda(const unsigned char* input, unsigned char* output,
                   int width, int height, int channels,
                   const float* host_kernel, int ksize)
{
    size_t img_bytes = (size_t)width * height * channels * sizeof(unsigned char);
    size_t kernel_bytes = (size_t)ksize * ksize * sizeof(float);

    unsigned char *d_in=nullptr, *d_out=nullptr;
    float *d_kernel=nullptr;

    CHECK_CUDA(cudaMalloc(&d_in, img_bytes));
    CHECK_CUDA(cudaMalloc(&d_out, img_bytes));
    CHECK_CUDA(cudaMalloc(&d_kernel, kernel_bytes));

    CHECK_CUDA(cudaMemcpy(d_in, input, img_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, host_kernel, kernel_bytes, cudaMemcpyHostToDevice));

    const int TILE_X = 16, TILE_Y = 16;
    dim3 block(TILE_X, TILE_Y);
    dim3 grid((width + TILE_X - 1) / TILE_X, (height + TILE_Y - 1) / TILE_Y);

    int half = ksize/2;
    int tileW = TILE_X + 2*half;
    int tileH = TILE_Y + 2*half;
    int shared_bytes = channels * tileW * tileH * sizeof(unsigned char);

    conv_shared_kernel<<<grid, block, shared_bytes>>>(d_in, d_out, width, height, channels, d_kernel, ksize);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(output, d_out, img_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_kernel);
}
