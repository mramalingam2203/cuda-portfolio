#pragma once
#ifndef KERNELS_CUH
#define KERNELS_CUH

// Host-callable wrapper to produce a blended frame on the GPU.
//
// d_frameA, d_frameB: device pointers to grayscale frames (unsigned char),
// d_out: device pointer to output grayscale frame (unsigned char).
// width, height: frame dimensions.
// alpha: interpolation factor in [0,1], e.g. 0.5 -> midway
void launchBlendFrames(const unsigned char* d_frameA,
                       const unsigned char* d_frameB,
                       unsigned char* d_out,
                       int width, int height,
                       float alpha);

#endif // KERNELS_CUH
