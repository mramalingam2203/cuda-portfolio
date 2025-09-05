#ifndef MONTECARLO_CUH
#define MONTECARLO_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

// Kernel to initialize RNG states
__global__ void initCurandPhilox(curandStatePhilox4_32_10_t *state, unsigned long seed, int n);

// Kernel to simulate multi-step GBM paths
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
);

#endif // MONTECARLO_CUH
