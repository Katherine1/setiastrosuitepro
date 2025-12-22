#pragma once

#include <vector>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/core.hpp>

namespace py = pybind11;

class Stacker {
public:
    enum class Algo {
        MEAN,
        MEDIAN,
        SIGMA_CLIP,
        WINDSORIZED,
        TRIMMED_MEAN,
        ESD,
        BIWEIGHT,
        MODIFIED_Z_SCORE,
        MAX_VALUE
    };

    Stacker() = default;

    // Main entry point for processing a stack of pixels (chunk/tile)
    // Returns a tuple: (integrated_image, rejection_map)
    // stack: (F, H, W, C) float32 array
    // weights: (F,) or (F, H, W, C) float32 array
    static std::pair<py::array_t<float>, py::array_t<bool>> processStack(
        py::array_t<float> stack,
        py::array_t<float> weights,
        const std::string& algo_name,
        float sigma_low = 3.0,
        float sigma_high = 3.0,
        int iterations = 3,
        float kappa = 3.0,
        float trim_fraction = 0.1,
        float esd_threshold = 3.0,
        float biweight_constant = 6.0,
        float modz_threshold = 3.5,
        float max_val_threshold = 1.0 // Unused but consistent signature
    );

private:
    static Algo parseAlgo(const std::string& name);
    
    // Internal implementations
    static void algoMean(const float* stack_ptr, const float* weight_ptr, int F, int H, int W, int C, float* out_ptr);
    static void algoMedian(const float* stack_ptr, int F, int H, int W, int C, float* out_ptr);
    static void algoSigmaClip(const float* stack_ptr, const float* weight_ptr, int F, int H, int W, int C, float* out_ptr, bool* rej_ptr, float kappa, int iters);
    static void algoWindsorized(const float* stack_ptr, const float* weight_ptr, int F, int H, int W, int C, float* out_ptr, bool* rej_ptr, float low, float high, int iters);
    static void algoTrimmedMean(const float* stack_ptr, const float* weight_ptr, int F, int H, int W, int C, float* out_ptr, bool* rej_ptr, float trim_frac);
    static void algoESD(const float* stack_ptr, const float* weight_ptr, int F, int H, int W, int C, float* out_ptr, bool* rej_ptr, float threshold);
    // ... add others
};
