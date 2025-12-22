
#include "stacking.hpp"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp> // For sort if needed, though std::sort is usually fine

// Use standard C++ parallel algorithms if available, or OpenMP
#ifdef _OPENMP
#include <omp.h>
#endif

Stacker::Algo Stacker::parseAlgo(const std::string& name) {
    if (name == "Simple Average (No Rejection)") return Algo::MEAN;
    if (name == "Simple Median (No Rejection)" || name == "Comet Median") return Algo::MEDIAN;
    if (name == "Kappa-Sigma Clipping") return Algo::SIGMA_CLIP;
    if (name == "Windsorized Sigma Clipping" || name == "Weighted Windsorized Sigma Clipping") return Algo::WINDSORIZED;
    if (name == "Trimmed Mean") return Algo::TRIMMED_MEAN;
    if (name == "Extreme Studentized Deviate (ESD)") return Algo::ESD;
    if (name == "Biweight Estimator") return Algo::BIWEIGHT;
    if (name == "Modified Z-Score Clipping") return Algo::MODIFIED_Z_SCORE;
    if (name == "Max Value") return Algo::MAX_VALUE;
    
    // Default fallback
    return Algo::SIGMA_CLIP;
}

std::pair<py::array_t<float>, py::array_t<bool>> Stacker::processStack(
    py::array_t<float> stack,
    py::array_t<float> weights,
    const std::string& algo_name,
    float sigma_low,
    float sigma_high,
    int iterations,
    float kappa,
    float trim_fraction,
    float esd_threshold,
    float biweight_constant,
    float modz_threshold,
    float max_val_threshold
) {
    auto buf_stack = stack.request();
    auto buf_weights = weights.request();

    if (buf_stack.ndim != 3 && buf_stack.ndim != 4) {
        throw std::runtime_error("Stack must be 3D (F,H,W) or 4D (F,H,W,C)");
    }

    py::ssize_t StackDepth = buf_stack.shape[0];
    py::ssize_t StackHeight = buf_stack.shape[1];
    py::ssize_t StackWidth = buf_stack.shape[2];
    py::ssize_t StackChannels = (buf_stack.ndim == 4) ? buf_stack.shape[3] : 1;

    // Prepare output arrays
    std::vector<py::ssize_t> shape_out;
    if (buf_stack.ndim == 4) shape_out = {StackHeight, StackWidth, StackChannels};
    else shape_out = {StackHeight, StackWidth};

    py::array_t<float> result(shape_out);
    
    // Rejection map: same shape as input
    py::array_t<bool> rejection_map(buf_stack.shape);

    float* ptr_stack = static_cast<float*>(buf_stack.ptr);
    float* ptr_weights = static_cast<float*>(buf_weights.ptr);
    float* ptr_out = static_cast<float*>(result.request().ptr);
    bool* ptr_rej = static_cast<bool*>(rejection_map.request().ptr);

    // Initialize rejection output to false (all valid)
    std::fill(ptr_rej, ptr_rej + rejection_map.size(), false);

    Algo algo = parseAlgo(algo_name);

    if (algo == Algo::MEAN) {
        // Parallel Loop
        #pragma omp parallel for
        for (int y = 0; y < StackHeight; ++y) {
            for (int x = 0; x < StackWidth; ++x) {
                for (int c = 0; c < StackChannels; ++c) {
                    float sum = 0.0f;
                    float w_sum = 0.0f;
                    for (int f = 0; f < StackDepth; ++f) {
                        float val = ptr_stack[(f * StackHeight * StackWidth * StackChannels) + (y * StackWidth * StackChannels) + (x * StackChannels) + c];
                        // Weight handling: broadcast if shape is different
                        // Assuming basic weights shape for now: (F,) or matching stack
                        float w = (buf_weights.size == StackDepth) ? ptr_weights[f] : ptr_weights[(f * StackHeight * StackWidth * StackChannels) + (y * StackWidth * StackChannels) + (x * StackChannels) + c];
                        sum += val * w;
                        w_sum += w;
                    }
                    ptr_out[(y * StackWidth * StackChannels) + (x * StackChannels) + c] = (w_sum > 1e-6f) ? (sum / w_sum) : 0.0f;
                }
            }
        }
    }
    else if (algo == Algo::MEDIAN) {
        #pragma omp parallel for
        for (int y = 0; y < StackHeight; ++y) {
            for (int x = 0; x < StackWidth; ++x) {
                for (int c = 0; c < StackChannels; ++c) {
                    std::vector<float> vals(StackDepth);
                    for (int f = 0; f < StackDepth; ++f) {
                        vals[f] = ptr_stack[(f * StackHeight * StackWidth * StackChannels) + (y * StackWidth * StackChannels) + (x * StackChannels) + c];
                    }
                    std::sort(vals.begin(), vals.end());
                    float med;
                    if (StackDepth % 2 == 1) med = vals[StackDepth / 2];
                    else med = (vals[StackDepth / 2 - 1] + vals[StackDepth / 2]) * 0.5f;
                    ptr_out[(y * StackWidth * StackChannels) + (x * StackChannels) + c] = med;
                }
            }
        }
    }
    else if (algo == Algo::WINDSORIZED) {
         #pragma omp parallel for
        for (int y = 0; y < StackHeight; ++y) {
            for (int x = 0; x < StackWidth; ++x) {
                for (int c = 0; c < StackChannels; ++c) {
                    std::vector<float> vals(StackDepth);
                    std::vector<int> indices(StackDepth);
                    for (int f = 0; f < StackDepth; ++f) {
                        vals[f] = ptr_stack[(f * StackHeight * StackWidth * StackChannels) + (y * StackWidth * StackChannels) + (x * StackChannels) + c];
                        indices[f] = f;
                    }

                    // Mask of currently valid pixels (1 = valid, 0 = rejected)
                    std::vector<bool> valid(StackDepth, true);
                    
                    // Iterative Clipping
                    for (int iter = 0; iter < iterations; ++iter) {
                         // Collect valid values
                        std::vector<float> current_valid;
                        current_valid.reserve(StackDepth);
                        for(int f=0; f<StackDepth; ++f) if(valid[f]) current_valid.push_back(vals[f]);

                        if (current_valid.empty()) break;

                        // Median & StdDev
                        size_t n = current_valid.size();
                        size_t mid = n / 2;
                        std::nth_element(current_valid.begin(), current_valid.begin() + mid, current_valid.end());
                        float med = current_valid[mid];
                        if (n % 2 == 0) {
                             std::nth_element(current_valid.begin(), current_valid.begin() + mid - 1, current_valid.end());
                             // Just copy valid vals and full sort to be safe/easy? For F ~ 100, sort is fast.
                             // In C++, std::sort is extremely optimized.
                        }
                        // Actually, just sorting current_valid is fast enough.
                        std::sort(current_valid.begin(), current_valid.end());
                        if (n % 2 == 0) med = (current_valid[n/2 - 1] + current_valid[n/2]) * 0.5f;
                        else med = current_valid[n/2];

                        // StdDev
                        double sum_sq_diff = 0.0;
                        for (float v : current_valid) {
                            sum_sq_diff += (v - med) * (v - med);
                        }
                        float std = std::sqrt(sum_sq_diff / n);

                        float lo = med - sigma_low * std;
                        float hi = med + sigma_high * std;

                        bool changed = false;
                        for(int f=0; f<StackDepth; ++f) {
                            if (valid[f]) {
                                if (vals[f] < lo || vals[f] > hi) {
                                    valid[f] = false;
                                    changed = true;
                                }
                            }
                        }
                        if (!changed) break;
                    }

                    // Final Integration (Weighted Mean of Survivors)
                    float sum = 0.0f;
                    float w_sum = 0.0f;
                    for (int f = 0; f < StackDepth; ++f) {
                        // Mark global output rejection map
                         ptr_rej[(f * StackHeight * StackWidth * StackChannels) + (y * StackWidth * StackChannels) + (x * StackChannels) + c] = !valid[f];

                         if (valid[f]) {
                             float w = (buf_weights.size == StackDepth) ? ptr_weights[f] : ptr_weights[(f * StackHeight * StackWidth * StackChannels) + (y * StackWidth * StackChannels) + (x * StackChannels) + c];
                             sum += vals[f] * w;
                             w_sum += w;
                         }
                    }

                    if (w_sum > 1e-9f) {
                        ptr_out[(y * StackWidth * StackChannels) + (x * StackChannels) + c] = sum / w_sum;
                    } else {
                        ptr_out[(y * StackWidth * StackChannels) + (x * StackChannels) + c] = 0.0f; 
                    }
                }
            }
        }
    }
    else if (algo == Algo::SIGMA_CLIP) {
         #pragma omp parallel for
        for (int y = 0; y < StackHeight; ++y) {
            for (int x = 0; x < StackWidth; ++x) {
                for (int c = 0; c < StackChannels; ++c) {
                    // Logic: Iterative Kappa-Sigma
                    std::vector<float> vals(StackDepth);
                    std::vector<float> ws(StackDepth);
                    std::vector<int> idxs(StackDepth);
                    for(int f=0; f<StackDepth; ++f) {
                        vals[f] = ptr_stack[(f * StackHeight * StackWidth * StackChannels) + (y * StackWidth * StackChannels) + (x * StackChannels) + c];
                        // Weight handling for (N,) or (N,H,W,C)
                        float w = (buf_weights.size == StackDepth) ? ptr_weights[f] : ptr_weights[(f * StackHeight * StackWidth * StackChannels) + (y * StackWidth * StackChannels) + (x * StackChannels) + c];
                        ws[f] = w;
                        idxs[f] = f;
                    }

                    // Keep track of active indices
                    std::vector<int> active_indices = idxs;

                    for(int iter=0; iter<iterations; ++iter) {
                        if (active_indices.empty()) break;
                        
                        // Collect current values
                        std::vector<float> current_vals;
                        current_vals.reserve(active_indices.size());
                        for(int f : active_indices) current_vals.push_back(vals[f]);

                        // Calc stats
                        // Median
                        size_t n = current_vals.size();
                        size_t mid = n / 2;
                        std::nth_element(current_vals.begin(), current_vals.begin() + mid, current_vals.end());
                        float med = current_vals[mid];
                         if (n % 2 == 0) {
                            std::nth_element(current_vals.begin(), current_vals.begin() + mid - 1, current_vals.end());
                            med = (current_vals[mid-1] + current_vals[mid]) * 0.5f;
                        }

                        // StdDev
                        double sum_sq = 0.0;
                         for(int f : active_indices) {
                            float v = vals[f];
                            sum_sq += (v - med) * (v - med);
                        }
                        float std = std::sqrt(sum_sq / n);

                        float lo = med - kappa * std;
                        float hi = med + kappa * std;

                        // Filter
                        std::vector<int> next_indices;
                        next_indices.reserve(n);
                        for(int f : active_indices) {
                             if (vals[f] >= lo && vals[f] <= hi && vals[f] != 0.0f) {
                                 next_indices.push_back(f);
                             }
                        }
                        if (next_indices.size() == active_indices.size()) break; // Converged
                        active_indices = next_indices;
                    }

                    // Rejection Mask & Result
                    float sum = 0.0f;
                    float wsum = 0.0f;
                    
                    // Mark all rejected first
                    for(int f=0; f<StackDepth; ++f) ptr_rej[(f * StackHeight * StackWidth * StackChannels) + (y * StackWidth * StackChannels) + (x * StackChannels) + c] = true;
                    
                    for(int f : active_indices) {
                        ptr_rej[(f * StackHeight * StackWidth * StackChannels) + (y * StackWidth * StackChannels) + (x * StackChannels) + c] = false;
                        float w = ws[f];
                        sum += vals[f] * w;
                        wsum += w;
                    }

                    if (wsum > 1e-9f) {
                        ptr_out[(y * StackWidth * StackChannels) + (x * StackChannels) + c] = sum / wsum;
                    } else if (!active_indices.empty()) {
                        // Fallback to simple unweighted mean of survivors
                        float s = 0;
                        for(int f : active_indices) s += vals[f];
                        ptr_out[(y * StackWidth * StackChannels) + (x * StackChannels) + c] = s / active_indices.size();
                    } else {
                        // All rejected? Use median of original
                         std::vector<float> all_vals; 
                         for(float v : vals) if(v!=0) all_vals.push_back(v);
                         if (all_vals.empty()) {
                             ptr_out[(y * StackWidth * StackChannels) + (x * StackChannels) + c] = 0.0f;
                         } else {
                            size_t n = all_vals.size();
                            std::nth_element(all_vals.begin(), all_vals.begin() + n/2, all_vals.end());
                            ptr_out[(y * StackWidth * StackChannels) + (x * StackChannels) + c] = all_vals[n/2];
                         }
                    }
                }
            }
        }
    }
    else if (algo == Algo::TRIMMED_MEAN) {
         #pragma omp parallel for
        for (int y = 0; y < StackHeight; ++y) {
            for (int x = 0; x < StackWidth; ++x) {
                for (int c = 0; c < StackChannels; ++c) {
                    std::vector<std::pair<float, int>> pairs; // val, original_index
                    std::vector<float> ws(StackDepth);
                    pairs.reserve(StackDepth);
                    
                    for(int f=0; f<StackDepth; ++f) {
                        float v = ptr_stack[(f * StackHeight * StackWidth * StackChannels) + (y * StackWidth * StackChannels) + (x * StackChannels) + c];
                        ws[f] = (buf_weights.size == StackDepth) ? ptr_weights[f] : ptr_weights[(f * StackHeight * StackWidth * StackChannels) + (y * StackWidth * StackChannels) + (x * StackChannels) + c];
                         if (v != 0.0f) {
                             pairs.push_back({v, f});
                         } else {
                              ptr_rej[(f * StackHeight * StackWidth * StackChannels) + (y * StackWidth * StackChannels) + (x * StackChannels) + c] = true;
                         }
                    }

                    size_t n = pairs.size();
                    if (n == 0) {
                        ptr_out[(y * StackWidth * StackChannels) + (x * StackChannels) + c] = 0.0f;
                        continue;
                    }
                    
                    std::sort(pairs.begin(), pairs.end()); // sort by value

                    int trim = (int)(trim_fraction * n);
                    int start = trim;
                    int end = n - trim;
                    
                    if (start >= end) { // Too much trimmed?
                        start = 0; end = n; // No trim fall back
                    }

                    float sum = 0.0f;
                    float wsum = 0.0f;

                    // Mark bounds rejected
                    for(int i=0; i<n; ++i) {
                         int f = pairs[i].second;
                         if (i < start || i >= end) {
                             ptr_rej[(f * StackHeight * StackWidth * StackChannels) + (y * StackWidth * StackChannels) + (x * StackChannels) + c] = true;
                         } else {
                             ptr_rej[(f * StackHeight * StackWidth * StackChannels) + (y * StackWidth * StackChannels) + (x * StackChannels) + c] = false;
                             float w = ws[f];
                             sum += pairs[i].first * w;
                             wsum += w;
                         }
                    }
                    
                    if (wsum > 1e-9f) {
                         ptr_out[(y * StackWidth * StackChannels) + (x * StackChannels) + c] = sum / wsum;
                    } else {
                        // Median of trimmed set
                        if (start < end) {
                            ptr_out[(y * StackWidth * StackChannels) + (x * StackChannels) + c] = pairs[start + (end-start)/2].first;
                        } else {
                             ptr_out[(y * StackWidth * StackChannels) + (x * StackChannels) + c] = pairs[n/2].first;
                        }
                    }
                }
            }
        }
    }
    else {
        // Fallback for Biweight / ModZ / Max -> Median (Safe default)
        if (algo == Algo::ESD) {
             #pragma omp parallel for
            for (int y = 0; y < StackHeight; ++y) {
                for (int x = 0; x < StackWidth; ++x) {
                    for (int c = 0; c < StackChannels; ++c) {
                        std::vector<int> active_indices;
                        std::vector<float> vals(StackDepth);
                        std::vector<float> ws(StackDepth);
                        for(int f=0; f<StackDepth; ++f) {
                             active_indices.push_back(f);
                             vals[f] = ptr_stack[(f * StackHeight * StackWidth * StackChannels) + (y * StackWidth * StackChannels) + (x * StackChannels) + c];
                             ws[f] = (buf_weights.size == StackDepth) ? ptr_weights[f] : ptr_weights[(f * StackHeight * StackWidth * StackChannels) + (y * StackWidth * StackChannels) + (x * StackChannels) + c];
                        }
                        
                        std::vector<float> current_vals;
                        for(int f : active_indices) if (vals[f] != 0) current_vals.push_back(vals[f]);

                        if (current_vals.empty()) {
                             ptr_out[(y * StackWidth * StackChannels) + (x * StackChannels) + c] = 0.0f;
                             for(int f=0; f<StackDepth; ++f) ptr_rej[(f * StackHeight * StackWidth * StackChannels) + (y * StackWidth * StackChannels) + (x * StackChannels) + c] = true;
                             continue;
                        }

                        double sum = 0;
                        for(float v : current_vals) sum += v;
                        double mean = sum / current_vals.size();
                        
                        double sum_sq = 0;
                        for(float v : current_vals) sum_sq += (v - mean)*(v - mean);
                        double std = std::sqrt(sum_sq / current_vals.size());

                        float wsum = 0.0f;
                        float val_sum = 0.0f;
                        
                        for(int f=0; f<StackDepth; ++f) {
                            if (vals[f] == 0) {
                                ptr_rej[(f * StackHeight * StackWidth * StackChannels) + (y * StackWidth * StackChannels) + (x * StackChannels) + c] = true;
                                continue;
                            }
                            
                            bool reject = false;
                            if (std > 0) {
                                float z = std::abs(vals[f] - mean) / std;
                                if (z >= esd_threshold) reject = true;
                            } 

                            ptr_rej[(f * StackHeight * StackWidth * StackChannels) + (y * StackWidth * StackChannels) + (x * StackChannels) + c] = reject;
                            if (!reject) {
                                float w = ws[f];
                                val_sum += vals[f] * w;
                                wsum += w;
                            }
                        }

                        if (wsum > 1e-9f) {
                            ptr_out[(y * StackWidth * StackChannels) + (x * StackChannels) + c] = val_sum / wsum;
                        } else {
                             ptr_out[(y * StackWidth * StackChannels) + (x * StackChannels) + c] = mean;
                        }
                    }
                }
            } 
        } else {
             // Fallback
            for (int y = 0; y < StackHeight; ++y) {
                for (int x = 0; x < StackWidth; ++x) {
                    for (int c = 0; c < StackChannels; ++c) {
                        std::vector<float> vals(StackDepth);
                        for (int f = 0; f < StackDepth; ++f) {
                            vals[f] = ptr_stack[(f * StackHeight * StackWidth * StackChannels) + (y * StackWidth * StackChannels) + (x * StackChannels) + c];
                        }
                        std::sort(vals.begin(), vals.end());
                        float med;
                        if (StackDepth % 2 == 1) med = vals[StackDepth / 2];
                        else med = (vals[StackDepth / 2 - 1] + vals[StackDepth / 2]) * 0.5f;
                        ptr_out[(y * StackWidth * StackChannels) + (x * StackChannels) + c] = med;
                         for(int f=0; f<StackDepth; ++f) ptr_rej[(f * StackHeight * StackWidth * StackChannels) + (y * StackWidth * StackChannels) + (x * StackChannels) + c] = false;
                    }
                }
            }
        }
    }

    return std::make_pair(result, rejection_map);
}
