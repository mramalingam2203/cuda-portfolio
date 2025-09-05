#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "kernels.cuh"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input_video> <output_video> <num_inbetweens>\n";
        std::cerr << "Example: ./video_interp input.mp4 output.mp4 3  # inserts 3 frames between each pair\n";
        return -1;
    }

    std::string inputPath = argv[1];
    std::string outputPath = argv[2];
    int num_inbetweens = std::max(0, std::atoi(argv[3])); // number of interpolated frames per pair

    cv::VideoCapture cap(inputPath);
    if (!cap.isOpened()) {
        std::cerr << "Error: cannot open input file: " << inputPath << "\n";
        return -1;
    }

    int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0;

    // We write color frames (BGR). We'll convert grayscale interpolation results back to BGR.
    int fourcc = cv::VideoWriter::fourcc('m','p','4','v'); // cross-platform MP4
    cv::VideoWriter writer(outputPath, fourcc, fps, cv::Size(width, height), true);
    if (!writer.isOpened()) {
        std::cerr << "Error: cannot open output file: " << outputPath << "\n";
        return -1;
    }

    cv::Mat frameA_color, frameB_color, frameA_gray, frameB_gray;
    if (!cap.read(frameA_color)) {
        std::cerr << "Error: input contains no frames\n";
        return -1;
    }
    cv::cvtColor(frameA_color, frameA_gray, cv::COLOR_BGR2GRAY);

    // allocate device buffers once
    size_t frameBytes = static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(unsigned char);
    unsigned char *d_A = nullptr, *d_B = nullptr, *d_out = nullptr;
    if (cudaMalloc(&d_A, frameBytes) != cudaSuccess ||
        cudaMalloc(&d_B, frameBytes) != cudaSuccess ||
        cudaMalloc(&d_out, frameBytes) != cudaSuccess) {
        std::cerr << "Error: cudaMalloc failed\n";
        return -1;
    }

    // process pairs
    while (true) {
        if (!cap.read(frameB_color)) break;
        cv::cvtColor(frameB_color, frameB_gray, cv::COLOR_BGR2GRAY);

        // write original frame A (color)
        writer.write(frameA_color);

        // copy frameA_gray and frameB_gray to device
        cudaMemcpy(d_A, frameA_gray.data, frameBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, frameB_gray.data, frameBytes, cudaMemcpyHostToDevice);

        // generate in-between frames
        for (int i = 1; i <= num_inbetweens; ++i) {
            float alpha = float(i) / float(num_inbetweens + 1); // evenly spaced between 0 and 1
            launchBlendFrames(d_A, d_B, d_out, width, height, alpha);

            // copy back
            cv::Mat interpGray(height, width, CV_8UC1);
            cudaMemcpy(interpGray.data, d_out, frameBytes, cudaMemcpyDeviceToHost);

            // convert to color and write
            cv::Mat interpColor;
            cv::cvtColor(interpGray, interpColor, cv::COLOR_GRAY2BGR);
            writer.write(interpColor);
        }

        // advance: next pair
        frameA_color = frameB_color.clone();
        frameA_gray  = frameB_gray.clone();
    }

    // write final frame
    writer.write(frameA_color);

    // cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_out);
    cap.release();
    writer.release();

    std::cout << "Done. Output: " << outputPath << "\n";
    std::cout << "Inserted " << num_inbetweens << " frames between each original pair (output is slower/longer).\n";
    return 0;
}
