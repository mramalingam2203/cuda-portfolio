#include "montecarlo.cuh"
#include <curand_kernel.h>
#include <math.h>

__global__ void initCurandPhilox(curandStatePhilox4_32_10_t *state, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &state[idx]);
    }
}

__global__ void gbmPathsMultiStep(
    curandStatePhilox4_32_10_t *state,
    float *d_results,
    float *d_paths,
    int n,
    int steps,
    float S0,
    float mu,
    float sigma,
    float T,
    bool storePaths
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandStatePhilox4_32_10_t localState = state[idx];
        float dt = T / steps;
        float price = S0;

        // Simulate GBM path
        for (int i = 0; i < steps; i++) {
            float randNorm = curand_normal(&localState);
            price *= expf((mu - 0.5f * sigma * sigma) * dt + sigma * sqrtf(dt) * randNorm);
            if (storePaths) {
                d_paths[idx * steps + i] = price;
            }
        }

        d_results[idx] = price; // Final price
        state[idx] = localState;
    }
}
