#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "montecarlo.cuh"

#define THREADS_PER_BLOCK 256

int main() {
    int n = 1 << 18;    // ~262,144 simulations
    int steps = 252;    // trading days in a year
    float S0 = 100.0f, mu = 0.05f, sigma = 0.2f, T = 1.0f;

    size_t resultsSize = n * sizeof(float);
    size_t pathsSize = n * steps * sizeof(float);

    float *d_results;
    cudaMalloc(&d_results, resultsSize);

    float *d_paths;
    cudaMalloc(&d_paths, pathsSize);

    curandStatePhilox4_32_10_t *d_state;
    cudaMalloc(&d_state, n * sizeof(curandStatePhilox4_32_10_t));

    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Initialize RNG states
    initCurandPhilox<<<blocks, THREADS_PER_BLOCK>>>(d_state, time(NULL), n);
    cudaDeviceSynchronize();

    // Run multi-step GBM simulation
    gbmPathsMultiStep<<<blocks, THREADS_PER_BLOCK>>>(
        d_state, d_results, d_paths,
        n, steps, S0, mu, sigma, T, true
    );
    cudaDeviceSynchronize();

    // Copy back final prices
    std::vector<float> h_results(n);
    cudaMemcpy(h_results.data(), d_results, resultsSize, cudaMemcpyDeviceToHost);

    // Compute average price
    long double sum = 0.0;
    for (auto &val : h_results) sum += val;
    std::cout << "Estimated final price (mean): " << sum / n << std::endl;

    cudaFree(d_results);
    cudaFree(d_paths);
    cudaFree(d_state);

    return 0;
}
