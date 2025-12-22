#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace saspro {

class Deconvolver {
public:
    // Main entry points
    static cv::Mat richardsonLucy(const cv::Mat& img, const cv::Mat& psf, int iterations, bool use_tv_reg=false, float tv_weight=0.002f);
    static cv::Mat wiener(const cv::Mat& img, const cv::Mat& psf, float snr);
    static cv::Mat vanCittert(const cv::Mat& img, const cv::Mat& psf, int iterations, float relaxation);

private:
    // Helper for tiled FFT convolution if needed, though for now we might focus on direct FFT
    // since Deconvolution usually fits in memory for single band.
    // If not, we'll need a TiledDeconv wrapper.
    
    // Helper to compute FFT
    static void computeDFT(const cv::Mat& src, cv::Mat& dst);
    static void computeIDFT(const cv::Mat& src, cv::Mat& dst);
    
    // Regularization
    static cv::Mat computeTVRegularization(const cv::Mat& img, float weight);
};

} // namespace saspro
