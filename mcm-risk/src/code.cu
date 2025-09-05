// mc_gbm.cu
// Compile: nvcc -O3 mc_gbm.cu -o mc_gbm -lcurand
// Run example: ./mc_gbm 1000000 252 42
//
// This program:
//  - uses cuRAND (Philox) to initialize RNG states per path
//  - simulates GBM paths (one path per thread) using grid-stride loop
//  - writes final returns to device array
//  - sorts returns using Thrust on device to compute VaR and ES
//  - prints timings and VaR/ES at 95% and 99%

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_CURAND(call) do { \
    curandStatus_t cs = (call); \
    if (cs != CURAND_STATUS_SUCCESS) { \
        fprintf(stderr, "cuRAND error %s:%d: %d\n", __FILE__, __LINE__, cs); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Simple host-side clipped to ensure we don't allocate insane amounts in examples
inline size_t clamp_sz(size_t x, size_t lo, size_t hi){ return (x < lo) ? lo : (x > hi ? hi : x); }

// Kernel: initialize curand philox states (one per path)
__global__ void init_rng(curandStatePhilox4_32_10_t* states, unsigned long long seed, int num_paths) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < num_paths; i += stride) {
        // subsequence = i ensures independent streams per thread
        curand_init(seed, (unsigned long long)i, 0ULL, &states[i]);
    }
}

// Kernel: simulate GBM paths; one thread simulates many steps for a single path
// Inputs:
//   mu, sigma: scalars (we keep single-asset for clarity; extend to arrays for multi-asset)
//   S0: initial price
//   steps: # timesteps
//   dt: timestep size
//   num_paths: total number of paths
// Outputs:
//   out_returns[path] = (S_T - S0) / S0 (simple return)
__global__ void simulate_gbm_paths(float mu, float sigma, float S0,
                                   int steps, float dt,
                                   int num_paths,
                                   float* __restrict__ out_returns,
                                   curandStatePhilox4_32_10_t* states) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int path = tid; path < num_paths; path += stride) {
        curandStatePhilox4_32_10_t localState = states[path];
        float S = S0;
        // Use the Euler-Maruyama discretization for GBM:
        // S_{t+dt} = S_t * exp((mu - 0.5 sigma^2) dt + sigma * sqrt(dt) * Z)
        float drift = (mu - 0.5f * sigma * sigma) * dt;
        float vol_scale = sigma * sqrtf(dt);

        // Loop over time steps
        for (int t = 0; t < steps; ++t) {
            float z = curand_normal(&localState); // standard normal
            S = S * expf(drift + vol_scale * z);
        }

        // store return
        out_returns[path] = (S - S0) / S0;

        // write back state
        states[path] = localState;
    }
}

int main(int argc, char** argv) {
    // Arguments: <num_paths> <steps> <seed>
    if (argc < 2) {
        printf("Usage: %s <num_paths> [steps=252] [seed=42]\n", argv[0]);
        return 0;
    }

    // Parse args
    const long long num_paths = atoll(argv[1]);
    const int steps = (argc > 2) ? atoi(argv[2]) : 252;
    const unsigned long long seed = (argc > 3) ? strtoull(argv[3], NULL, 10) : 42ULL;

    if (num_paths <= 0) {
        fprintf(stderr, "num_paths must be > 0\n");
        return 1;
    }

    // Simulation parameters (example)
    const float mu = 0.05f;      // annual drift
    const float sigma = 0.2f;    // annual vol
    const float S0 = 100.0f;
    const float T_years = 1.0f;
    const float dt = T_years / steps;

    printf("MC GPU GBM: paths=%lld steps=%d seed=%llu\n", num_paths, steps, seed);

    // Device allocations
    float* d_returns = nullptr;
    curandStatePhilox4_32_10_t* d_states = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&d_returns, sizeof(float) * (size_t)num_paths));
    CHECK_CUDA(cudaMalloc((void**)&d_states, sizeof(curandStatePhilox4_32_10_t) * (size_t)num_paths));

    // Launch config (tune to your GPU)
    int threads = 256;
    long long blocks = (num_paths + threads - 1) / threads;
    if (blocks > 65535) {
        // use a reasonable upper bound for grid.x and use grid-stride; we can use 2D grid, but keep simple:
        blocks = 65535;
    }
    dim3 blockDim(threads);
    dim3 gridDim((int)blocks);

    // Init RNG
    printf("Initializing RNG states (grid=%d block=%d)\n", gridDim.x, blockDim.x);

    // Time using CUDA events
    cudaEvent_t ev_start, ev_after_init, ev_after_sim;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_after_init));
    CHECK_CUDA(cudaEventCreate(&ev_after_sim));
    CHECK_CUDA(cudaEventRecord(ev_start, 0));

    init_rng<<<gridDim, blockDim>>>(d_states, seed, (int)0);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(ev_after_init, 0));

    // simulate
    simulate_gbm_paths<<<gridDim, blockDim>>>(mu, sigma, S0, steps, dt, (int)num_paths, d_returns, d_states);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(ev_after_sim, 0));

    // synchronize and measure
    CHECK_CUDA(cudaEventSynchronize(ev_after_sim));
    float ms_init = 0.0f, ms_sim = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_init, ev_start, ev_after_init));
    CHECK_CUDA(cudaEventElapsedTime(&ms_sim, ev_after_init, ev_after_sim));
    printf("Time: RNG init = %.3f ms, simulation = %.3f ms\n", ms_init, ms_sim);

    // Compute VaR and ES (on device using Thrust)
    // We will sort the returns ascending. VaR at confidence level c (e.g., 0.95) is the (1-c) quantile of losses.
    // Here returns = profit fraction. For losses, we take negative returns.
    // For example VaR95 = - quantile(0.05) if quantile is return level.

    // Copy or operate on device pointer via thrust
    thrust::device_ptr<float> d_ptr(d_returns);
    // sort ascending
    printf("Sorting %lld returns on device (this may be expensive for very large N)...\n", num_paths);
    thrust::sort(thrust::device, d_ptr, d_ptr + num_paths);

    // compute VaR at 95% and 99%:
    auto get_var_es = [&](double conf) {
        // conf = 0.95 -> alpha = 1 - conf = 0.05
        double alpha = 1.0 - conf;
        // index (0-based) for quantile
        long long idx = (long long)floor(alpha * (double)num_paths);
        if (idx < 0) idx = 0;
        if (idx >= num_paths) idx = num_paths - 1;
        // copy that element from device
        float ret_at_idx = 0.0f;
        CHECK_CUDA(cudaMemcpy(&ret_at_idx, d_returns + idx, sizeof(float), cudaMemcpyDeviceToHost));

        // Expected Shortfall (ES) = average of losses worse than VaR -> average of returns[0..idx]
        double es = 0.0;
        if (idx > 0) {
            // allocate a small host buffer if idx not too big; otherwise do device reduction
            // We'll compute ES on device using thrust::reduce on range
            thrust::device_ptr<float> begin = d_ptr;
            thrust::device_ptr<float> end = d_ptr + idx + 1;
            double sum = thrust::reduce(thrust::device, begin, end, (double)0.0);
            es = sum / (double)(idx + 1);
        } else {
            es = ret_at_idx;
        }

        // VaR returned as negative of returns quantile (loss positive). But we'll present VaR as positive loss.
        double var = - (double) ret_at_idx;
        double es_loss = - es;
        return std::make_pair(var, es_loss);
    };

    auto v95 = get_var_es(0.95);
    auto v99 = get_var_es(0.99);

    printf("VaR95 = %.6f   ES95 = %.6f\n", v95.first, v95.second);
    printf("VaR99 = %.6f   ES99 = %.6f\n", v99.first, v99.second);

    // Cleanup
    CHECK_CUDA(cudaFree(d_returns));
    CHECK_CUDA(cudaFree(d_states));
    CHECK_CUDA(cudaEventDestroy(ev_start));
    CHECK_CUDA(cudaEventDestroy(ev_after_init));
    CHECK_CUDA(cudaEventDestroy(ev_after_sim));

    return 0;
}

