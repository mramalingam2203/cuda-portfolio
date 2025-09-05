#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cstdio>

// 2D kernel: per-pixel linear blend (grayscale)
__global__ static void blendKernel(const unsigned char* frameA,
                                   const unsigned char* frameB,
                                   unsigned char* out,
                                   int width, int height,
                                   float alpha) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float a = static_cast<float>(frameA[idx]);
    float b = static_cast<float>(frameB[idx]);
    float v = (1.0f - alpha) * a + alpha * b;
    out[idx] = static_cast<unsigned char>(v + 0.5f);
}

void launchBlendFrames(const unsigned char* d_frameA,
                       const unsigned char* d_frameB,
                       unsigned char* d_out,
                       int width, int height,
                       float alpha) {
    const int BX = 16;
    const int BY = 16;
    dim3 block(BX, BY);
    dim3 grid((width + BX - 1) / BX, (height + BY - 1) / BY);

    blendKernel<<<grid, block>>>(d_frameA, d_frameB, d_out, width, height, alpha);
    cudaError_t cerr = cudaGetLastError();
    if (cerr != cudaSuccess) {
        std::fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(cerr));
    }
    // synchronize to ensure result is ready for the host copy
    cudaDeviceSynchronize();
}
