
#include "background_extractor.hpp"
#include <omp.h>
#include <cmath>
#include <iostream>

namespace saspro {

// --- Polynomial Utils ---

// Number of coeffs for degree D: (D+1)(D+2)/2
static int numCoeffs(int degree) {
    return (degree + 1) * (degree + 2) / 2;
}

std::vector<double> BackgroundExtractor::fitPolynomial(const std::vector<Point>& points, int degree) {
    int N = (int)points.size();
    int M = numCoeffs(degree);
    
    if (N < M) {
        // Not enough points
        return std::vector<double>(M, 0.0);
    }

    // Solve A * x = B
    // A is NxM, x is Mx1, B is Nx1
    cv::Mat A(N, M, CV_64F);
    cv::Mat B(N, 1, CV_64F);

    for (int i = 0; i < N; i++) {
        double px = points[i].x; // normalize? Usually helpful if coords are large.
        double py = points[i].y;
        double val = points[i].value;
        
        // Populate row of A
        int col = 0;
        for (int d = 0; d <= degree; d++) {
            for (int k = 0; k <= d; k++) {
                int x_exp = d - k;
                int y_exp = k;
                A.at<double>(i, col++) = std::pow(px, x_exp) * std::pow(py, y_exp);
            }
        }
        B.at<double>(i, 0) = val;
    }

    cv::Mat x;
    bool success = cv::solve(A, B, x, cv::DECOMP_SVD);
    
    std::vector<double> coeffs(M);
    if (success) {
        for (int i = 0; i < M; i++) {
            coeffs[i] = x.at<double>(i, 0);
        }
    }
    return coeffs;
}

cv::Mat BackgroundExtractor::evaluatePolynomial(int width, int height, const std::vector<double>& coeffs, int degree) {
    cv::Mat model(height, width, CV_32F);
    
    // Precompute powers? Or just iterate. 
    // Degree is small usually (1..4).
    
    // Optimization: normalize coords to [-1, 1] if fitting did, but we used raw.
    // We must stick to what fit used. If fit used raw pixels, we use raw pixels.
    // Note: Raw powers of 4000 can overflow double if not careful with scale, or lose precision.
    // Python implementation normally just does lstsq on raw coords.
    
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        float* row = model.ptr<float>(y);
        for (int x = 0; x < width; x++) {
            double val = 0.0;
            int col = 0;
            // Matches order in fitPolynomial
            for (int d = 0; d <= degree; d++) {
                for (int k = 0; k <= d; k++) {
                    int x_exp = d - k;
                    int y_exp = k;
                    val += coeffs[col++] * std::pow((double)x, x_exp) * std::pow((double)y, y_exp);
                }
            }
            row[x] = (float)val;
        }
    }
    return model;
}

// --- RBF Utils ---

// Basis function: Multiquadric sqrt(r^2 + epsilon^2) or Inverse. 
// ABE usually uses multiquadric or thin-plate. 
// Python scipy defaults to multiquadric: sqrt((r/epsilon)^2 + 1)
// Let's use basic r^2 log r (Thin Plate) or Multiquadric.
// We'll use Multiquadric: phi(r) = sqrt(r^2 + smooth^2)

static inline double rbf_kernel(double r2, double smooth2) {
    return std::sqrt(r2 + smooth2);
}

cv::Mat BackgroundExtractor::generateRBFModel(int width, int height, const std::vector<Point>& points, float smoothing) {
    int N = (int)points.size();
    if (N == 0) return cv::Mat::zeros(height, width, CV_32F);

    // 1. Solve weights
    // Matrix G (NxN), G_ij = phi(|pi - pj|)
    // Weights w = G^-1 * values
    
    double s2 = (double)smoothing * (double)smoothing;
    cv::Mat G(N, N, CV_64F);
    cv::Mat B(N, 1, CV_64F);
    
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        B.at<double>(i, 0) = points[i].value;
        for (int j = 0; j < N; j++) {
            double dx = points[i].x - points[j].x;
            double dy = points[i].y - points[j].y;
            double r2 = dx*dx + dy*dy;
            G.at<double>(i, j) = rbf_kernel(r2, s2);
        }
    }
    
    cv::Mat w;
    cv::solve(G, B, w, cv::DECOMP_SVD); // SVD is robust for RBF matrices
    
    std::vector<double> weights(N);
    for(int i=0; i<N; i++) weights[i] = w.at<double>(i, 0);
    
    // 2. Evaluate grid
    // F(x,y) = sum( w_i * phi(|x - xi|) )
    cv::Mat model(height, width, CV_32F);
    
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        float* row = model.ptr<float>(y);
        for (int x = 0; x < width; x++) {
            double sum = 0.0;
            for (int i = 0; i < N; i++) {
                double dx = x - points[i].x;
                double dy = y - points[i].y;
                double r2 = dx*dx + dy*dy;
                sum += weights[i] * rbf_kernel(r2, s2);
            }
            row[x] = (float)sum;
        }
    }
    
    return model;
}

// --- Sample Generation ---

std::vector<BackgroundExtractor::Point> BackgroundExtractor::generateSamplePoints(const cv::Mat& img, int box_size, int grid_step, float sigma_clip) {
    std::vector<Point> samples;
    
    // Iterate grid
    int Height = img.rows;
    int Width = img.cols;
    int hk = box_size / 2;
    
    for (int y = hk; y < Height - hk; y += grid_step) {
        for (int x = hk; x < Width - hk; x += grid_step) {
            // Region
            cv::Rect roi(x - hk, y - hk, box_size, box_size);
            cv::Mat patch = img(roi);
            
            // Calc stats
            cv::Scalar mean, stddev; 
            cv::meanStdDev(patch, mean, stddev);
            
            // Sigma clip logic (simple single pass or double)
            // Python ABE uses smart exclusion. Simple approx:
            // Median of box is a good robust estimate of background if sparsity of stars is high.
            // But we want "darkest" background.
            // Let's compute Median of the patch.
            
            // Sorting entire patch is slow.
            // Simple robust estimator:
            // If patch has stars (high variance), mean > median.
            // We want the background. 
            // Let's take the iter/sigma clipped mean.
            
            // Optimization: Copy to vector, N partial sort (median)
            // Or just use mean if stddev is low.
            
            // Fast approximation for C++:
            // 1. Mean/Std
            // 2. Reject > Mean + 1.0*Std
            // 3. Recalc Mean
            
            float m = (float)mean[0];
            float s = (float)stddev[0];
            
            float sum = 0;
            int count = 0;
            float thresh = m + sigma_clip * s; 
            
            // Iterate patch pixels
            // This is inner loop, optimize access
            if (patch.isContinuous()) {
               const float* p = patch.ptr<float>();
               int total = box_size*box_size;
               for(int k=0; k<total; k++) {
                   if (p[k] < thresh) {
                       sum += p[k];
                       count++;
                   }
               }
            } else {
                for(int r=0; r<patch.rows; r++) {
                    const float* ptr = patch.ptr<float>(r);
                    for(int c=0; c<patch.cols; c++) {
                        if (ptr[c] < thresh) {
                            sum += ptr[c];
                            count++;
                        }
                    }
                }
            }
            
            if (count > 0) {
                samples.push_back({(float)x, (float)y, sum / count});
            }
        }
    }
    
    return samples;
}

} // namespace saspro
