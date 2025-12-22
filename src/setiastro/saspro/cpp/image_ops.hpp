#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

namespace saspro {

class ImageOps {
public:
    static cv::Mat blendImages(const cv::Mat& A, const cv::Mat& B, const std::string& mode, float opacity);
    
    static cv::Mat flipImage(const cv::Mat& img, int axis); // axis: 0=vertical, 1=horizontal
    static cv::Mat rotateImage(const cv::Mat& img, int flag); // 0=90CW, 1=90CCW, 2=180
    static cv::Mat invertImage(const cv::Mat& img);

    // In-place calibration
    // img: input/output
    // dark: optional dark frame (can be empty)
    // flat: optional flat frame (can be empty)
    // bias: optional bias frame (can be empty)
    // pedestal: scalar to add after subtraction
    static void calibrateImage(cv::Mat& img, const cv::Mat& dark, const cv::Mat& flat, const cv::Mat& bias, float pedestal);
    
    // Resize with high quality
    static cv::Mat rescaleImage(const cv::Mat& img, float factor);
};

} // namespace saspro
