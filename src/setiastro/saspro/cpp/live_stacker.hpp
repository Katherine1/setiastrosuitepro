#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace saspro {

// A stateful stacker for Live Stacking
class LiveStacker {
public:
    LiveStacker();
    ~LiveStacker();

    void reset();
    
    // Add a frame to the running stack
    // align_transform: 2x3 affine matrix (float). passed as vector or Mat?
    // If aligned frame is provided directly, caller handles alignment.
    void addFrame(const cv::Mat& image);
    
    // Get current stacked result
    // sigma_clip: calculate sigma from M2 and clip 'last' frame? 
    // Wait, Live Stacking usually outputs the 'Mean' of all frames.
    // For Sigma Clipping, we need history or robust integration.
    // Welford gives Mean and StdDev. We can't retrospectively sigma-clip without history.
    // But we CAN output: (Mean) or (Mean of Valid).
    // The Python code: 
    //  - Bootstraps (linear avg) until N frames
    //  - Then uses Welford Mean + Sigma rejection on *incoming* frame?
    //  No, "Mode: μ-σ Clipping Average" in Python usually implies checking new frame against current stats
    //  and rejecting it if outlier, OR updating stats with it.
    
    // We'll mimic the Python logic:
    // Update Mean/M2.
    // Return Mean.
    
    cv::Mat getMean();
    cv::Mat getCount(); // Per-pixel count if we implement rejection
    
    // Get M2/Variance/Sigma for analysis
    cv::Mat getSigma();
    
    int getFrameCount() const { return frame_count_; }

    // Rejection Parameters
    void setSigmaClip(float threshold) { sigma_threshold_ = threshold; }

private:
    int frame_count_;
    cv::Mat accum_mean_;
    cv::Mat accum_m2_;
    
    float sigma_threshold_; // If > 0, we reject pixels from update? 
    // Welford update with rejection:
    // 1. Calc current Sigma = sqrt(M2 / (k-1))
    // 2. If |x - Mean| > thresh * Sigma, skip update for this pixel
    // 3. Else update Mean, M2
};

} // namespace saspro
