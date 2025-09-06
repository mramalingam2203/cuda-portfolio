#include <opencv2/opencv.hpp>
#include "grayscale.cuh"
#include "colorspace.cuh"
#include "gamma_brightness.cuh"

int main() {
    
    // RGB to Grayscale
    cv::Mat img = cv::imread("../data/input.png", cv::IMREAD_COLOR);
    if (img.empty()) {
        printf("Image not found!\n");
        return -1;
    }

    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();

    cv::Mat gray(height, width, CV_8UC1);

    rgb_to_grayscale_cuda(img.data, gray.data, width, height, channels);

    cv::imwrite("../output/gray.png", gray);


    //Color Space Conversion Engine
    
    
    cv::Mat ycbcr(height, width, CV_8UC3);
    cv::Mat hsv(height, width, CV_8UC3);

    rgb_to_ycbcr_cuda(img.data, ycbcr.data, width, height, channels);
    rgb_to_hsv_cuda(img.data, hsv.data, width, height, channels);

    cv::imwrite("../output/ycbcr.png", ycbcr);
    cv::imwrite("../output/hsv.png", hsv);


    // Gamma & Brightness correction engine

    cv::Mat bright(height, width, CV_8UC3);
    cv::Mat gamma_corrected(height, width, CV_8UC3);

    // Brightness scaling: alpha=1.2 (contrast), beta=30 (brightness shift)
    brightness_scale_cuda(img.data, bright.data, width, height, channels, 1.2f, 30.0f);

    // Gamma correction: gamma=0.8 (brighten)
    gamma_correction_cuda(img.data, gamma_corrected.data, width, height, channels, 0.8f);

    cv::imwrite("../output/bright.png", bright);
    cv::imwrite("../output/gamma.png", gamma_corrected);


    return 0;
    
}
