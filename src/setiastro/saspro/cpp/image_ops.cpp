#include "image_ops.hpp"
#include <opencv2/core/hal/hal.hpp>
#include <cmath>
#include <omp.h>

namespace saspro {

cv::Mat ImageOps::blendImages(const cv::Mat& A, const cv::Mat& B, const std::string& mode, float opacity) {
    CV_Assert(A.size() == B.size());
    CV_Assert(A.type() == B.type());
    CV_Assert(A.depth() == CV_32F); // Assume float32 for now

    cv::Mat out = A.clone();
    int channels = A.channels();
    int rows = A.rows;
    int cols = A.cols * channels; 

    if (A.isContinuous() && B.isContinuous() && out.isContinuous()) {
        cols *= rows;
        rows = 1;
    }

    const float* pA = A.ptr<float>();
    const float* pB = B.ptr<float>();
    float* pOut = out.ptr<float>();

    // opacity: 0.0 = A, 1.0 = Result. v = A*(1-alpha) + B*alpha
    // But most modes define B as the 'blend' layer and A as 'base'. 
    // The Numba code implemented: v = A*(1-alpha) + f(A,B)*alpha
    
    // We can dispatch based on mode string once
    enum Mode { NORMAL, ADD, SUBTRACT, MULTIPLY, SCREEN, OVERLAY, DIFFERENCE, DIVIDE };
    Mode m = NORMAL;
    if (mode == "add") m = ADD;
    else if (mode == "subtract") m = SUBTRACT;
    else if (mode == "multiply") m = MULTIPLY;
    else if (mode == "screen") m = SCREEN;
    else if (mode == "overlay") m = OVERLAY;
    else if (mode == "difference") m = DIFFERENCE;
    else if (mode == "divide") m = DIVIDE;

    #pragma omp parallel for
    for (int i = 0; i < cols; i++) {
        float a = pA[i];
        float b_in = pB[i];
        float b_out = b_in; // default for normal

        switch (m) {
            case ADD:
                b_out = a + b_in;
                break;
            case SUBTRACT:
                b_out = a - b_in;
                break;
            case MULTIPLY:
                b_out = a * b_in; // or a * (1-alpha) + (a*b)*alpha ?? Numba was: (A*(1-a) + A*B*a) which is interpolation
                // Wait, typical multiply blend is A*B.
                // The Numba logic was: v = A * (1-alpha) + (A*B) * alpha
                // So 'b_out' here represents the target value at 100% opacity.
                b_out = a * b_in; 
                break;
            case SCREEN:
                // 1 - (1-a)*(1-b)
                b_out = 1.0f - (1.0f - a) * (1.0f - b_in);
                break;
            case DIVIDE:
                b_out = a / (b_in + 1e-6f);
                break;
            case DIFFERENCE:
                b_out = std::abs(a - b_in);
                break;
            case OVERLAY:
                if (a <= 0.5f) {
                    b_out = 2.0f * a * b_in;
                } else {
                    b_out = 1.0f - 2.0f * (1.0f - a) * (1.0f - b_in);
                }
                break;
            default: // Normal
                b_out = b_in; // This replaces A with B at high opacity
                break;
        }
        
        // Clamp result of blend function
        if (b_out < 0.0f) b_out = 0.0f;
        if (b_out > 1.0f) b_out = 1.0f;

        // Mix with original A
        // if mode != subtract: v = A*(1-alpha) + b_out*alpha
        // For subtract, numba was A - B*alpha. 
        // Let's stick to the Numba logic strictly if possible, or standard blending.
        // Numba Subtract: v = A - B*alpha.
        
        float v;
        if (m == SUBTRACT) {
             v = a - b_in * opacity;
        } else if (m == ADD) {
             // Numba Add: v = A + B*alpha
             v = a + b_in * opacity;
        } else if (m == MULTIPLY) {
             // Numba Multiply: v = A*(1-alpha) + (A*B)*alpha
             // Which matches standard lerp(A, A*B, alpha)
             v = a * (1.0f - opacity) + (a * b_in) * opacity;
        } else {
             // Standard lerp for others
             v = a * (1.0f - opacity) + b_out * opacity;
        }

        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        
        pOut[i] = v;
    }

    return out;
}

cv::Mat ImageOps::flipImage(const cv::Mat& img, int axis) {
    cv::Mat out;
    cv::flip(img, out, axis); // axis 0=vert, 1=horiz
    return out;
}

cv::Mat ImageOps::rotateImage(const cv::Mat& img, int flag) {
    cv::Mat out;
    // flag: 0=90CW, 1=90CCW, 2=180
    if (flag == 2) {
        cv::rotate(img, out, cv::ROTATE_180);
    } else if (flag == 0) {
        cv::rotate(img, out, cv::ROTATE_90_CLOCKWISE);
    } else {
        cv::rotate(img, out, cv::ROTATE_90_COUNTERCLOCKWISE);
    }
    return out;
}

cv::Mat ImageOps::invertImage(const cv::Mat& img) {
    // 1.0 - img
    cv::Mat out;
    cv::subtract(cv::Scalar::all(1.0), img, out);
    return out;
}

void ImageOps::calibrateImage(cv::Mat& img, const cv::Mat& dark, const cv::Mat& flat, const cv::Mat& bias, float pedestal) {
    // Basic idea: (img - dark - bias) / flat + pedestal
    // But dark usually includes bias if not optimized. 
    // LiveStacking logic: img = img - master_dark
    // Then img = img / master_flat
    
    // We'll support flexible accumulation.
    
    if (!dark.empty()) {
        CV_Assert(img.size() == dark.size());
        cv::subtract(img, dark, img);
    }
    
    if (!bias.empty()) {
        CV_Assert(img.size() == bias.size());
        cv::subtract(img, bias, img);
    }

    if (pedestal != 0.0f) {
        cv::add(img, cv::Scalar::all(pedestal), img);
    }

    if (!flat.empty()) {
        // Flat division: img /= (flat / mean(flat))
        // => img *= (mean(flat) / flat)
        // Optimize: compute scale factor = mean(flat)
        // Then img = img * scale / flat
        
        cv::Scalar mean_flat = cv::mean(flat);
        double avg = mean_flat[0];
        // If color flat, usually we normalize per channel?
        // Numba utils: median_flat = np.mean(master_flat) (scalar mean of whole array)
        // Let's stick to per-channel division if 3 channels, or global if 1.
        
        // Actually, efficiently:
        // img = img / flat * avg
        
        cv::divide(img, flat, img);
        cv::multiply(img, cv::Scalar::all(avg), img);
    }
}

cv::Mat ImageOps::rescaleImage(const cv::Mat& img, float factor) {
    cv::Mat out;
    cv::resize(img, out, cv::Size(), factor, factor, cv::INTER_LINEAR); // Linear is good/fast
    return out;
}

} // namespace saspro
