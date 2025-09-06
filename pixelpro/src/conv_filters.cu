#include "convolution.cuh"
#include <cmath>
#include <cstring>

// 3x3 Sobel X and Y
static const float SOBEL_GX[9] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
};
static const float SOBEL_GY[9] = {
     1,  2,  1,
     0,  0,  0,
    -1, -2, -1
};

// Prewitt
static const float PREWITT_GX[9] = {
    -1, 0, 1,
    -1, 0, 1,
    -1, 0, 1
};
static const float PREWITT_GY[9] = {
     1,  1,  1,
     0,  0,  0,
    -1, -1, -1
};

// Laplacian (4-neighbor)
static const float LAPLACIAN_4[9] = {
     0,  1,  0,
     1, -4,  1,
     0,  1,  0
};

// Helper: compute gradient magnitude from Gx and Gy host buffers (float)
static void compute_magnitude_uchar(const float* gx, const float* gy, unsigned char* out, int width, int height) {
    int N = width * height;
    for (int i = 0; i < N; ++i) {
        float mag = sqrtf(gx[i]*gx[i] + gy[i]*gy[i]);
        int v = (int)(mag + 0.5f);
        if (v > 255) v = 255;
        if (v < 0) v = 0;
        out[i] = (unsigned char)v;
    }
}

// For simplicity we will run two convolutions (Gx and Gy) using the generic convolve_cuda
// and compute magnitude on host. This is memory-copy heavy but simplest to reason about.
// For production you'd implement a single kernel computing both and writing magnitude.

void sobel_magnitude_cuda(const unsigned char* input, unsigned char* output_gray,
                          int width, int height, int channels)
{
    // allocate host floats for Gx and Gy results per pixel (single-channel)
    int N = width * height;
    float* gx = new float[N];
    float* gy = new float[N];
    unsigned char* tmpGx = new unsigned char[N * channels]; // use generic conv to produce channel-aligned outputs
    unsigned char* tmpGy = new unsigned char[N * channels];

    // run generic conv for Gx and Gy
    convolve_cuda(input, tmpGx, width, height, channels, SOBEL_GX, 3);
    convolve_cuda(input, tmpGy, width, height, channels, SOBEL_GY, 3);

    // convert to float single-channel luminance-like (if input is RGB, we can take channel 0 or compute grayscale)
    // We'll compute gray value from RGB of tmpGx/tmpGy using luminosity if channels==3
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx_pix = y*width + x;
            int idx = idx_pix * channels;
            float valx = 0.0f, valy = 0.0f;
            if (channels == 1) {
                valx = (float)tmpGx[idx];
                valy = (float)tmpGy[idx];
            } else { // RGB: compute perceived gray of the gradient results
                // linear combination (same as rgb2gray weights)
                valx = 0.299f*tmpGx[idx] + 0.587f*tmpGx[idx+1] + 0.114f*tmpGx[idx+2];
                valy = 0.299f*tmpGy[idx] + 0.587f*tmpGy[idx+1] + 0.114f*tmpGy[idx+2];
            }
            gx[idx_pix] = valx;
            gy[idx_pix] = valy;
        }
    }

    compute_magnitude_uchar(gx, gy, output_gray, width, height);

    delete[] gx; delete[] gy; delete[] tmpGx; delete[] tmpGy;
}

void prewitt_magnitude_cuda(const unsigned char* input, unsigned char* output_gray,
                            int width, int height, int channels)
{
    // similar to sobel but use PREWITT_GX / PREWITT_GY
    int N = width * height;
    float* gx = new float[N];
    float* gy = new float[N];
    unsigned char* tmpGx = new unsigned char[N * channels];
    unsigned char* tmpGy = new unsigned char[N * channels];

    convolve_cuda(input, tmpGx, width, height, channels, PREWITT_GX, 3);
    convolve_cuda(input, tmpGy, width, height, channels, PREWITT_GY, 3);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx_pix = y*width + x;
            int idx = idx_pix * channels;
            float valx = 0.0f, valy = 0.0f;
            if (channels == 1) {
                valx = (float)tmpGx[idx];
                valy = (float)tmpGy[idx];
            } else {
                valx = 0.299f*tmpGx[idx] + 0.587f*tmpGx[idx+1] + 0.114f*tmpGx[idx+2];
                valy = 0.299f*tmpGy[idx] + 0.587f*tmpGy[idx+1] + 0.114f*tmpGy[idx+2];
            }
            gx[idx_pix] = valx;
            gy[idx_pix] = valy;
        }
    }
    compute_magnitude_uchar(gx, gy, output_gray, width, height);

    delete[] gx; delete[] gy; delete[] tmpGx; delete[] tmpGy;
}

void laplacian_cuda(const unsigned char* input, unsigned char* output_gray,
                    int width, int height, int channels)
{
    // single convolution with Laplacian kernel; then convert to gray if input is RGB
    unsigned char* tmpOut = new unsigned char[width * height * channels];
    convolve_cuda(input, tmpOut, width, height, channels, LAPLACIAN_4, 3);

    // convert to single channel output_gray
    if (channels == 1) {
        memcpy(output_gray, tmpOut, width * height);
    } else {
        for (int i = 0; i < width * height; ++i) {
            int idx = i * channels;
            int v = (int)(0.299f*tmpOut[idx] + 0.587f*tmpOut[idx+1] + 0.114f*tmpOut[idx+2] + 0.5f);
            if (v < 0) v = 0; if (v > 255) v = 255;
            output_gray[i] = (unsigned char)v;
        }
    }

    delete[] tmpOut;
}
