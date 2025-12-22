#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace saspro {

class BackgroundExtractor {
public:
    struct Point {
        float x, y, value;
    };

    // Polynomial fitting
    // Returns coefficients
    static std::vector<double> fitPolynomial(const std::vector<Point>& points, int degree);
    
    // Evaluate polynomial model on full image
    static cv::Mat evaluatePolynomial(int width, int height, const std::vector<double>& coeffs, int degree);

    // RBF Interpolation (Thin Plate Spline or MultiQuadric)
    // Returns the background model directly
    static cv::Mat generateRBFModel(int width, int height, const std::vector<Point>& points, float smoothing);
    
    // Helper to generate sample points from image (grid-based with sigma clip rejection)
    // returns list of points (x, y, bg_value)
    static std::vector<Point> generateSamplePoints(const cv::Mat& img, int box_size, int grid_step, float sigma_clip);
};

} // namespace saspro
