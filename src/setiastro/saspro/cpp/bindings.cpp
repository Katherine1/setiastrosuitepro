
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
#include <algorithm>
#include <iostream>
#include <map>
#include <tuple>
#include "stacking.hpp"
#include "image_ops.hpp"
#include "deconvolver.hpp"
#include "background_extractor.hpp"
#include "live_stacker.hpp"

namespace py = pybind11;

// Placeholder for the Aligner class
class StarAligner {
public:
    StarAligner() {}

    // Structure to hold star info
    struct Star {
        float x, y;
        float flux;
    };

    /**
     * @brief Detect stars in an image.
     * 
     * @param image Input image (grayscale). CV_8U or CV_32F.
     * @param max_stars Maximum number of stars to return (brightest first).
     * @param detection_sigma Threshold sigma above mean background.
     * @param min_area Minimum pixel area for a star.
     * @return std::vector<cv::Point2f> List of star coordinates.
     */
    std::vector<cv::Point2f> detectStars(const py::array& input_image, int max_stars = 2000, double detection_sigma = 5.0, int min_area = 5) {
        // Convert numpy array to cv::Mat
        py::buffer_info buf = input_image.request();
        int type = -1;
        if (buf.format == py::format_descriptor<uint8_t>::format()) type = CV_8U;
        else if (buf.format == py::format_descriptor<float>::format()) type = CV_32F;
        else if (buf.format == py::format_descriptor<double>::format()) type = CV_64F;
        
        if(type == -1) throw std::runtime_error("Unsupported image type");

        cv::Mat img(buf.shape[0], buf.shape[1], type, buf.ptr);
        cv::Mat float_img;
        img.convertTo(float_img, CV_32F);

        // 1. Estimate background
        cv::Mat bg;
        // Using a box filter as a fast approximation for local median/background
        // Ideally should use a larger median blur or SEP's estimators, but this is fast and robust enough for bright stars
        int ksize = 15; 
        cv::blur(float_img, bg, cv::Size(ksize, ksize));

        // 2. Subtract background
        cv::Mat sub = float_img - bg;

        // 3. Global statistics for thresholding
        cv::Scalar mean, stddev;
        cv::meanStdDev(sub, mean, stddev);
        double threshold_val = mean[0] + detection_sigma * stddev[0];

        // 4. Threshold
        cv::Mat mask;
        cv::threshold(sub, mask, threshold_val, 255, cv::THRESH_BINARY);
        mask.convertTo(mask, CV_8U);

        // 5. Connected Components
        cv::Mat labels, stats, centroids;
        int nLabels = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);

        std::vector<Star> all_stars;
        all_stars.reserve(nLabels);

        for(int i = 1; i < nLabels; ++i) {
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if(area < min_area) continue;

            float cx = (float)centroids.at<double>(i, 0);
            float cy = (float)centroids.at<double>(i, 1);
            
            // Flux proxy: max value in the subtracted image at centroid
            // (More accurate would be sum of pixels in mask, but this suffices for sorting)
            float flux = 0.0f;
            int y_int = (int)cy;
            int x_int = (int)cx;
            if (y_int >= 0 && y_int < sub.rows && x_int >= 0 && x_int < sub.cols) {
                flux = sub.at<float>(y_int, x_int);
            }

            all_stars.push_back({cx, cy, flux});
        }

        // 6. Sort by flux descending
        std::sort(all_stars.begin(), all_stars.end(), [](const Star& a, const Star& b){
            return a.flux > b.flux;
        });

        // 7. Return top N
        std::vector<cv::Point2f> result;
        size_t count = std::min((size_t)max_stars, all_stars.size());
        result.reserve(count);
        for(size_t i = 0; i < count; ++i) {
            result.push_back(cv::Point2f(all_stars[i].x, all_stars[i].y));
        }

        return result;
    }
    // -------------------------------------------------------------------------
    // Pattern Matching Logic
    // -------------------------------------------------------------------------

    struct Triangle {
        int idx[3]; 
    };

    std::pair<float, float> computeInvariants(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3) {
        double d1 = cv::norm(p1 - p2);
        double d2 = cv::norm(p2 - p3);
        double d3 = cv::norm(p3 - p1);
        double sides[3] = {d1, d2, d3};
        std::sort(std::begin(sides), std::end(sides));
        
        if (sides[0] < 1e-6) return {0.0f, 0.0f};
        return {(float)(sides[1] / sides[0]), (float)(sides[2] / sides[0])};
    }

    // Quantize invariants for hashing
    // Using a simple integer key: (inv1 * 100, inv2 * 100) as in the python script
    std::pair<int, int> getHashKey(float inv1, float inv2, float tolerance = 0.01f) {
        // We round to nearest bin. 
        // 0.1 tolerance in python script is quite loose, we stick to that logic or refine it.
        // Python: inv_key = (round(inv[0], 2), round(inv[1], 2))
        // Floating point hash keys are tricky, so we convert to int.
        return {(int)std::round(inv1 * 100), (int)std::round(inv2 * 100)};
    }

    std::vector<Triangle> buildTriangles(const std::vector<cv::Point2f>& stars) {
        std::vector<Triangle> triangles;
        if (stars.size() < 3) return triangles;

        // Bounding rect
        cv::Rect2f rect;
        for (const auto& p : stars) {
            if (p.x < rect.x) rect.x = p.x;
            if (p.y < rect.y) rect.y = p.y;
            // logic above is flawed for initial point, easier to recompute properly
        }
        // Correct bounding box
        float minx = 1e9, miny = 1e9, maxx = -1e9, maxy = -1e9;
        for (const auto& p : stars) {
            if (p.x < minx) minx = p.x;
            if (p.y < miny) miny = p.y;
            if (p.x > maxx) maxx = p.x;
            if (p.y > maxy) maxy = p.y;
        }
        // Expand slightly to avoid boundary issues
        rect = cv::Rect2f(minx - 10, miny - 10, (maxx - minx) + 20, (maxy - miny) + 20);

        cv::Subdiv2D subdiv(rect);
        for (const auto& p : stars) {
            subdiv.insert(p);
        }

        std::vector<cv::Vec6f> triangleList;
        subdiv.getTriangleList(triangleList);

        for (const auto& t : triangleList) {
            cv::Point2f pt[3];
            pt[0] = cv::Point2f(t[0], t[1]);
            pt[1] = cv::Point2f(t[2], t[3]);
            pt[2] = cv::Point2f(t[4], t[5]);

            // Check if points are within image boundaries (Subdiv2D adds giant outer triangles)
            if (!rect.contains(pt[0]) || !rect.contains(pt[1]) || !rect.contains(pt[2]))
                continue;

            // Find indices in original star list
            // This brute-force search is slow O(N), but N is small (points per triangle) * M (triangles)
            // A map or KDTree would be faster, but for < 2000 stars, it's typically fine.
            // Optimization: The points returned by subdiv might not be EXACT binary matches to input floats.
            // But they should be very close.
            int idx[3] = {-1, -1, -1};
            for (int k = 0; k < 3; ++k) {
                for (size_t s = 0; s < stars.size(); ++s) {
                    if (cv::norm(stars[s] - pt[k]) < 0.1) {
                        idx[k] = (int)s;
                        break;
                    }
                }
            }
            if (idx[0] != -1 && idx[1] != -1 && idx[2] != -1) {
                triangles.push_back({idx[0], idx[1], idx[2]});
            }
        }
        return triangles;
    }

    /**
     * @brief Find affine transform between source (this) and reference (other).
     * 
     * @param src_points Source stars.
     * @param ref_points Reference stars.
     * @return py::tuple (matrix_2x3_numpy, success_bool)
     */
    py::tuple findTransform(const std::vector<cv::Point2f>& src_stars, 
                            const std::vector<cv::Point2f>& ref_stars,
                            bool use_homography = false) {
        
        // 1. Build triangles
        auto src_tris = buildTriangles(src_stars);
        auto ref_tris = buildTriangles(ref_stars);

        // 2. Build Hash Map for Reference
        // Key: (inv1_int, inv2_int) -> list of triangle indices
        std::map<std::pair<int, int>, std::vector<int>> ref_map;
        for (size_t i = 0; i < ref_tris.size(); ++i) {
            const auto& t = ref_tris[i];
            auto inv = computeInvariants(ref_stars[t.idx[0]], ref_stars[t.idx[1]], ref_stars[t.idx[2]]);
            if (inv.first == 0) continue;
            auto key = getHashKey(inv.first, inv.second);
            ref_map[key].push_back((int)i);
        }

        // 3. Match
        std::vector<std::pair<int, int>> matches; // indices into src_tris, ref_tris
        int tolerance_bins = 1; // Check neighbors +/- 1 bin? Python uses rigid 0.1 tolerance on floats. 
        // 0.1 on floats ~ 10 units in our *100 int space. So we should search standard key +/- 10.
        // Actually, Python code: abs(inv_src[0] - inv_tgt[0]) < tol
        
        for (size_t i = 0; i < src_tris.size(); ++i) {
            const auto& t = src_tris[i];
            auto inv = computeInvariants(src_stars[t.idx[0]], src_stars[t.idx[1]], src_stars[t.idx[2]]);
            if (inv.first == 0) continue;
            
            int k1 = (int)std::round(inv.first * 100);
            int k2 = (int)std::round(inv.second * 100);
            
            // Search neighborhood
            // optimizing: only check exact matches first? 
            // The float tolerance is 0.1, which is HUGE. 0.1 * 100 = 10 bins.
            // We should iterate over keys. Since map is sorted, maybe lower_bound?
            // Doing a full scan of the map is too slow?
            // Let's iterate over the ref map... NO, map size is O(N).
            
            // Better: Iterate ref map and see if close to src?
            // For now, let's implement the 'close enough' check by simplified binning or brute force if needed.
            // Given the Python implementation iterates ALL src and ALL tgt keys, it is O(N_src * N_tgt).
            // We can do better by just iterating the map keys if N_ref < N_src or vice versa?
            
            // Actually, let's try direct lookups with a small window.
            // If tolerance is 0.1, that's range [val-0.1, val+0.1].
            // In our int domain, that's +/- 10.
            
            // Just iterate over the map. It's safe given N is small (~2000 stars -> ~6000 triangles?).
            // 6000 * 6000 is 36M ops, doable in C++.
            
            for (auto const& [ref_key, ref_indices] : ref_map) {
                if (std::abs(ref_key.first - k1) < 10 && std::abs(ref_key.second - k2) < 10) {
                    for (int r_idx : ref_indices) {
                        matches.push_back({(int)i, r_idx});
                    }
                }
            }
        }

        if (matches.empty()) {
             return py::make_tuple(py::none(), false);
        }

        // 4. RANSAC
        // We have matched triangles. Each match gives us 3 point pairs.
        // Collect all potential point pairs. 
        std::vector<cv::Point2f> p_src, p_ref;
        p_src.reserve(matches.size() * 3);
        p_ref.reserve(matches.size() * 3);

        for (const auto& m : matches) {
            const auto& t_src = src_tris[m.first];
            const auto& t_ref = ref_tris[m.second];
            
            // We need to correspond vertices.
            // The invariants are sorted sides. We need to match based on side lengths to match vertices.
            // vertices: A, B, C. Sides: c (AB), a (BC), b (CA).
            // Sorted sides: s1, s2, s3.
            // We can match vertices by the angle or by the adjacent sides.
            
            // Simplest: Try all 3 rotations?
            // Or sort vertices by internal angle?
            
            // Let's rely on the fact that computeInvariants sorts sides.
            // But we don't have the permutation stored.
            // Re-matching vertices:
            auto match_vertices = [&](const Triangle& tri, const std::vector<cv::Point2f>& stars) {
                // Return indices sorted by angle or something consistent
                // Actually, let's just find the vertex opposite to the smallest side, then medium, then large.
                int i0 = tri.idx[0], i1 = tri.idx[1], i2 = tri.idx[2];
                double d01 = cv::norm(stars[i0] - stars[i1]);
                double d12 = cv::norm(stars[i1] - stars[i2]);
                double d20 = cv::norm(stars[i2] - stars[i0]);
                
                // There are 3 sides. Smallest side is X. Vertices are P1, P2.
                // The vertex opposite is P3.
                // Let's identify vertices [v_small_opp, v_med_opp, v_large_opp] ??
                // Better: Sort sides. s1 < s2 < s3.
                // s1 is side P_a-P_b. Opposite is P_c.
                // s2 is ... opposite P_a.
                // s3 is ... opposite P_b.
                // So reliable ordering is: [Opposite(SmallestSide), Opposite(MediumSide), Opposite(LargestSide)]
                
                // struct Side { double len; int v_opp; };
                // std::vector<Side> s = { {d12, i0}, {d20, i1}, {d01, i2} };
                // std::sort(s...);
                // return { s[0].v_opp, s[1].v_opp, s[2].v_opp };
                
                struct S { double len; int opp; };
                std::vector<S> sides = { {d12, i0}, {d20, i1}, {d01, i2} };
                std::sort(sides.begin(), sides.end(), [](const S& a, const S& b){ return a.len < b.len; });
                return std::vector<int>{sides[0].opp, sides[1].opp, sides[2].opp};
            };
            
            auto v_src = match_vertices(t_src, src_stars);
            auto v_ref = match_vertices(t_ref, ref_stars);
            
            p_src.push_back(src_stars[v_src[0]]); p_ref.push_back(ref_stars[v_ref[0]]);
            p_src.push_back(src_stars[v_src[1]]); p_ref.push_back(ref_stars[v_ref[1]]);
            p_src.push_back(src_stars[v_src[2]]); p_ref.push_back(ref_stars[v_ref[2]]);
        }
        
        if (p_src.size() < 6) return py::make_tuple(py::none(), false);

        // Estimate Rigid/Affine/Homography
        cv::Mat mask;
        cv::Mat HomographyMat;
        if (use_homography) {
            HomographyMat = cv::findHomography(p_src, p_ref, cv::RANSAC, 5.0, mask);
        } else {
            HomographyMat = cv::estimateAffine2D(p_src, p_ref, mask, cv::RANSAC, 5.0);
        }

        if (HomographyMat.empty()) return py::make_tuple(py::none(), false);
        
        // Convert to numpy
        // H is CV_64F usually
        py::array_t<double> result = py::array_t<double>({HomographyMat.rows, HomographyMat.cols});
        auto r = result.mutable_unchecked<2>();
        for (int i = 0; i < HomographyMat.rows; i++)
            for (int j = 0; j < HomographyMat.cols; j++)
                r(i, j) = HomographyMat.at<double>(i, j);

        return py::make_tuple(result, true);
    }

    /**
     * @brief Warp image using provided transform.
     */
    py::array warpImage(const py::array& input_image, const py::array& transform, std::tuple<int, int> output_shape) {
         // Convert numpy array to cv::Mat
        py::buffer_info buf = input_image.request();
        int type = -1;
        if (buf.format == py::format_descriptor<uint8_t>::format()) type = CV_8U;
        else if (buf.format == py::format_descriptor<float>::format()) type = CV_32F;
        
        if(type == -1) throw std::runtime_error("Unsupported image type");
        
        cv::Mat img(buf.shape[0], buf.shape[1], type, buf.ptr);
        
        // Transform
        py::buffer_info t_buf = transform.request();
        cv::Mat M(t_buf.shape[0], t_buf.shape[1], CV_64F, t_buf.ptr);

        int h = std::get<0>(output_shape);
        int w = std::get<1>(output_shape);
        
        cv::Mat out;
        if (M.rows == 2) {
             cv::warpAffine(img, out, M, cv::Size(w, h), cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, 0);
        } else {
             cv::warpPerspective(img, out, M, cv::Size(w, h), cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, 0);
        }
        
        // Return
        if (type == CV_8U) {
             return py::array_t<uint8_t>({h, w}, out.data);
        } else {
             return py::array_t<float>({h, w}, (float*)out.data);
        }
    }

    // -------------------------------------------------------------------------
    // Polynomial Logic (Poly3, Poly4)
    // -------------------------------------------------------------------------
    // We need to fit: Src = Poly(Ref)
    // For Order 3: 1, x, y, x2, xy, y2, x3, x2y, xy2, y3 (10 terms)
    // Ref points (x,y) -> Design Matrix A. Src points (u,v) -> RHS b.
    // Solve A * Coeffs = b (for u and v separately).

    struct PolyModel {
        int order;
        std::vector<double> coeffs_x; // Coefficients for X output
        std::vector<double> coeffs_y; // Coefficients for Y output
    };

    /**
     * @brief Estimate polynomial transform coefficients.
     */
    py::tuple findPolynomialTransform(const std::vector<cv::Point2f>& src_stars,
                                      const std::vector<cv::Point2f>& ref_stars,
                                      int order) {
        
        // 1. Match stars first.
        // We can reuse the matching logic from findTransform.
        // Or refactor findTransform to return matches.
        // For simplicity/speed in this iteration, let's duplicate the basic matching call logic 
        // OR better: Assume the user calls this AFTER matching? 
        // No, the python interface expects "stars in -> transform out".
        // Let's call the internal matching helper (we need to refactor it out).
        
        // REFACTOR START: Extract matching logic
        auto src_tris = buildTriangles(src_stars);
        auto ref_tris = buildTriangles(ref_stars);
        
        std::map<std::pair<int, int>, std::vector<int>> ref_map;
        for (size_t i = 0; i < ref_tris.size(); ++i) {
            const auto& t = ref_tris[i];
            auto inv = computeInvariants(ref_stars[t.idx[0]], ref_stars[t.idx[1]], ref_stars[t.idx[2]]);
            if (inv.first == 0) continue;
            auto key = getHashKey(inv.first, inv.second);
            ref_map[key].push_back((int)i);
        }

        std::vector<std::pair<int, int>> matches; 
        for (size_t i = 0; i < src_tris.size(); ++i) {
            const auto& t = src_tris[i];
            auto inv = computeInvariants(src_stars[t.idx[0]], src_stars[t.idx[1]], src_stars[t.idx[2]]);
            if (inv.first == 0) continue;
            int k1 = (int)std::round(inv.first * 100);
            int k2 = (int)std::round(inv.second * 100);
            for (auto const& [ref_key, ref_indices] : ref_map) {
                if (std::abs(ref_key.first - k1) < 10 && std::abs(ref_key.second - k2) < 10) {
                    for (int r_idx : ref_indices) {
                        matches.push_back({(int)i, r_idx});
                    }
                }
            }
        }
        
        if (matches.empty()) return py::make_tuple(py::none(), false);

        std::vector<cv::Point2f> p_src, p_ref;
        p_src.reserve(matches.size() * 3);
        p_ref.reserve(matches.size() * 3);
        
        for (const auto& m : matches) {
            const auto& t_src = src_tris[m.first];
            const auto& t_ref = ref_tris[m.second];
            
             // Inline vertex matching logic (from before)
            // ... (Copied for brevity, simpler to share but duplication is faster to implement now)
            auto match_vertices = [&](const Triangle& tri, const std::vector<cv::Point2f>& stars) {
                int i0 = tri.idx[0], i1 = tri.idx[1], i2 = tri.idx[2];
                double d01 = cv::norm(stars[i0] - stars[i1]);
                double d12 = cv::norm(stars[i1] - stars[i2]);
                double d20 = cv::norm(stars[i2] - stars[i0]);
                struct S { double len; int opp; };
                std::vector<S> sides = { {d12, i0}, {d20, i1}, {d01, i2} };
                std::sort(sides.begin(), sides.end(), [](const S& a, const S& b){ return a.len < b.len; });
                return std::vector<int>{sides[0].opp, sides[1].opp, sides[2].opp};
            };
            auto v_src = match_vertices(t_src, src_stars);
            auto v_ref = match_vertices(t_ref, ref_stars);
            p_src.push_back(src_stars[v_src[0]]); p_ref.push_back(ref_stars[v_ref[0]]);
            p_src.push_back(src_stars[v_src[1]]); p_ref.push_back(ref_stars[v_ref[1]]);
            p_src.push_back(src_stars[v_src[2]]); p_ref.push_back(ref_stars[v_ref[2]]);
        }
        
        if (p_src.size() < 10) return py::make_tuple(py::none(), false);

        // 2. RANSAC to remove outliers first (using Affine as a robust pre-filter)
        cv::Mat mask;
        cv::estimateAffine2D(p_ref, p_src, mask, cv::RANSAC, 5.0); // Note: Ref -> Src
        
        std::vector<cv::Point2f> good_src, good_ref;
        for(int i=0; i<mask.rows; ++i) {
            if(mask.at<uchar>(i)) {
                good_src.push_back(p_src[i]);
                good_ref.push_back(p_ref[i]);
            }
        }
        
        // Need minimal points for polynomial
        int min_pts = (order == 3) ? 10 : 15;
        if(good_ref.size() < min_pts) return py::make_tuple(py::none(), false);

        // 3. Build Design Matrix A
        // Terms order 3: 1, x, y, x2, xy, y2, x3, x2y, xy2, y3
        int N = (int)good_ref.size();
        int M = (order == 3) ? 10 : 15;
        
        cv::Mat A(N, M, CV_64F);
        cv::Mat Bx(N, 1, CV_64F);
        cv::Mat By(N, 1, CV_64F);
        
        for(int i=0; i<N; ++i) {
            double x = good_ref[i].x;
            double y = good_ref[i].y;
            double u = good_src[i].x;
            double v = good_src[i].y;
            
            // Fill row
            int col = 0;
            // Degree 0
            A.at<double>(i, col++) = 1.0;
            // Degree 1
            A.at<double>(i, col++) = x;
            A.at<double>(i, col++) = y;
            // Degree 2
            A.at<double>(i, col++) = x*x;
            A.at<double>(i, col++) = x*y;
            A.at<double>(i, col++) = y*y;
            // Degree 3
            A.at<double>(i, col++) = x*x*x;
            A.at<double>(i, col++) = x*x*y;
            A.at<double>(i, col++) = x*y*y;
            A.at<double>(i, col++) = y*y*y;
            
            if (order == 4) { // Only if order 4
                 // Degree 4
                A.at<double>(i, col++) = x*x*x*x;
                A.at<double>(i, col++) = x*x*x*y;
                A.at<double>(i, col++) = x*x*y*y;
                A.at<double>(i, col++) = x*y*y*y;
                A.at<double>(i, col++) = y*y*y*y;
            }
            
            Bx.at<double>(i) = u;
            By.at<double>(i) = v;
        }
        
        // Solve using SVD or QR
        cv::Mat X_coeffs, Y_coeffs;
        bool ok = cv::solve(A, Bx, X_coeffs, cv::DECOMP_SVD);
        bool ok2= cv::solve(A, By, Y_coeffs, cv::DECOMP_SVD);
        
        if(!ok || !ok2) return py::make_tuple(py::none(), false);
        
        // Return coeffs as a dictionary or tuple
        // Let's return a list of doubles
        std::vector<double> cx, cy;
        X_coeffs.copyTo(cx);
        Y_coeffs.copyTo(cy);
        
        // Pack into a python object. We can use a dict.
        py::dict res;
        res["order"] = order;
        res["cx"] = cx;
        res["cy"] = cy;
        
        return py::make_tuple(res, true);
    }
    
    // Tiny helper to evaluate polynomial at a point
    // This is hot loop code, so we inline it or structure carefully 
    // but in parallel_for it is fine.

    py::array warpImagePolynomial(const py::array& input_image, const py::dict& poly_def, std::tuple<int, int> output_shape) {
        int order = poly_def["order"].cast<int>();
        std::vector<double> cx = poly_def["cx"].cast<std::vector<double>>();
        std::vector<double> cy = poly_def["cy"].cast<std::vector<double>>();
        
        int h = std::get<0>(output_shape);
        int w = std::get<1>(output_shape);
        
        // Create map maps
        cv::Mat map_x(h, w, CV_32F);
        cv::Mat map_y(h, w, CV_32F);
        
        // Parallelize map generation
        cv::parallel_for_(cv::Range(0, h), [&](const cv::Range& range){
            for(int r = range.start; r < range.end; ++r) {
                float* ptr_x = map_x.ptr<float>(r);
                float* ptr_y = map_y.ptr<float>(r);
                double y = (double)r;
                
                // Precompute powers of y? 
                double y2 = y*y; 
                double y3 = y2*y;
                double y4 = (order==4)? y3*y : 0;
                
                for(int c = 0; c < w; ++c) {
                    double x = (double)c;
                    double x2 = x*x; 
                    double x3 = x2*x;
                    double x4 = (order==4)? x3*x : 0;
                    
                    // Evaluate poly (Src = Poly(Ref))
                    // Terms: 1, x, y, x2, xy, y2, x3, x2y, xy2, y3...
                    
                    // X output
                    double u = cx[0] + cx[1]*x + cx[2]*y + 
                               cx[3]*x2 + cx[4]*x*y + cx[5]*y2 +
                               cx[6]*x3 + cx[7]*x2*y + cx[8]*x*y2 + cx[9]*y3;
                    if(order==4) {
                        u += cx[10]*x4 + cx[11]*x3*y + cx[12]*x2*y2 + cx[13]*x*y3 + cx[14]*y4;
                    }
                    
                    // Y output
                    double v = cy[0] + cy[1]*x + cy[2]*y + 
                               cy[3]*x2 + cy[4]*x*y + cy[5]*y2 +
                               cy[6]*x3 + cy[7]*x2*y + cy[8]*x*y2 + cy[9]*y3;
                    if(order==4) {
                        v += cy[10]*x4 + cy[11]*x3*y + cy[12]*x2*y2 + cy[13]*x*y3 + cy[14]*y4;
                    }
                    
                    ptr_x[c] = (float)u;
                    ptr_y[c] = (float)v;
                }
            }
        });
        
        // Remap
        py::buffer_info buf = input_image.request();
        int type = -1;
        if (buf.format == py::format_descriptor<uint8_t>::format()) type = CV_8U;
        else if (buf.format == py::format_descriptor<float>::format()) type = CV_32F;
        else if (buf.format == py::format_descriptor<double>::format()) type = CV_64F; // Remap supports? Converts to float usually.
        
        cv::Mat img(buf.shape[0], buf.shape[1], type, buf.ptr);
        cv::Mat out;
        
        cv::remap(img, out, map_x, map_y, cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, cv::Scalar(0));
        
        if (type == CV_8U) {
             return py::array_t<uint8_t>({h, w}, out.data);
        } else {
             return py::array_t<float>({h, w}, (float*)out.data);
        }
    }
};

#include "stacking.hpp"

PYBIND11_MODULE(saspro_cpp, m) {
    m.doc() = "SetiAstroSuite Pro C++ Backend (Star Alignment & Stacking)";

    py::class_<Stacker>(m, "Stacker")
        .def(py::init<>())
        .def_static("process_stack", &Stacker::processStack,
            py::arg("stack"),
            py::arg("weights"),
            py::arg("algo_name"),
            py::arg("sigma_low") = 3.0f,
            py::arg("sigma_high") = 3.0f,
            py::arg("iterations") = 3,
            py::arg("kappa") = 3.0f,
            py::arg("trim_fraction") = 0.1f,
            py::arg("esd_threshold") = 3.0f,
            py::arg("biweight_constant") = 6.0f,
            py::arg("modz_threshold") = 3.5f,
            py::arg("max_val_threshold") = 1.0f,
            "Process a stack of images (chunk) with rejection.");

    py::class_<StarAligner>(m, "StarAligner")
        .def(py::init<>())
        .def("detect_stars", &StarAligner::detectStars,
             py::arg("image"), py::arg("max_stars") = 500, py::arg("detection_sigma") = 3.0, py::arg("min_area") = 5,
             "Detect stars in the image.")
        .def("find_transform", &StarAligner::findTransform,
             py::arg("src_stars"), py::arg("dst_stars"), py::arg("src_shape"), py::arg("dst_shape"),
             "Find affine transform or homography.")
        .def("warp_image", &StarAligner::warpImage,
             py::arg("image"), py::arg("transform"), py::arg("output_shape"),
             "Warp image using the transform.")
        .def("find_polynomial_transform", &StarAligner::findPolynomialTransform,
            py::arg("src_stars"), py::arg("dst_stars"), py::arg("src_shape"), py::arg("dst_shape"), py::arg("order") = 3,
            "Find polynomial transform (order 3 or 4).")
        //.def("warp_image_polynomial", (py::array (StarAligner::*)(const py::array&, const py::dict&, std::tuple<int, int>)) &StarAligner::warpImagePolynomial,
        //    py::arg("image"), py::arg("poly_def"), py::arg("output_shape"),
        //    "Warp image using polynomial coefficients using geometric remap.");

    // --- ImageOps ---
    py::class_<saspro::ImageOps>(m, "ImageOps")
        .def_static("blend_images", &saspro::ImageOps::blendImages, py::arg("A"), py::arg("B"), py::arg("mode"), py::arg("opacity"))
        .def_static("flip_image", &saspro::ImageOps::flipImage, py::arg("img"), py::arg("axis"))
        .def_static("rotate_image", &saspro::ImageOps::rotateImage, py::arg("img"), py::arg("flag"))
        .def_static("invert_image", &saspro::ImageOps::invertImage, py::arg("img"))
        .def_static("calibrate_image", &saspro::ImageOps::calibrateImage, py::arg("img"), py::arg("dark")=cv::Mat(), py::arg("flat")=cv::Mat(), py::arg("bias")=cv::Mat(), py::arg("pedestal")=0.0f)
        .def_static("rescale_image", &saspro::ImageOps::rescaleImage, py::arg("img"), py::arg("factor"));

    // --- Deconvolver ---
    py::class_<saspro::Deconvolver>(m, "Deconvolver")
        .def_static("richardson_lucy", &saspro::Deconvolver::richardsonLucy, py::arg("img"), py::arg("psf"), py::arg("iterations"), py::arg("use_tv_reg")=false, py::arg("tv_weight")=0.002f)
        .def_static("wiener", &saspro::Deconvolver::wiener, py::arg("img"), py::arg("psf"), py::arg("snr"))
        .def_static("van_cittert", &saspro::Deconvolver::vanCittert, py::arg("img"), py::arg("psf"), py::arg("iterations"), py::arg("relaxation"));

    // --- BackgroundExtractor ---
    py::class_<saspro::BackgroundExtractor>(m, "BackgroundExtractor")
        .def(py::init<>())
        .def_static("fit_polynomial", &saspro::BackgroundExtractor::fitPolynomial, py::arg("points"), py::arg("degree"))
        .def_static("evaluate_polynomial", &saspro::BackgroundExtractor::evaluatePolynomial, py::arg("width"), py::arg("height"), py::arg("coeffs"), py::arg("degree"))
        .def_static("generate_rbf_model", [](int w, int h, py::array_t<float> pts, float smooth){
            py::buffer_info buf = pts.request();
            if (buf.ndim != 2 || buf.shape[1] != 3) throw std::runtime_error("Points must be Nx3 array");
            float* ptr = static_cast<float*>(buf.ptr);
            std::vector<saspro::BackgroundExtractor::Point> vec(buf.shape[0]);
            for(int i=0; i<buf.shape[0]; i++) {
                vec[i] = {ptr[i*3], ptr[i*3+1], ptr[i*3+2]};
            }
            return saspro::BackgroundExtractor::generateRBFModel(w, h, vec, smooth);
        }, py::arg("width"), py::arg("height"), py::arg("points"), py::arg("smoothing"))
        .def_static("generate_sample_points", [](const cv::Mat& img, int box, int step, float sigma){
             auto pts = saspro::BackgroundExtractor::generateSamplePoints(img, box, step, sigma);
             std::vector<py::ssize_t> shape = {(py::ssize_t)pts.size(), 3};
             py::array_t<float> ret(shape);
             auto r = ret.mutable_unchecked<2>();
             for(py::ssize_t i=0; i<(py::ssize_t)pts.size(); i++) {
                 r(i,0) = pts[i].x; r(i,1) = pts[i].y; r(i,2) = pts[i].value;
             }
             return ret;
        }, py::arg("img"), py::arg("box_size"), py::arg("grid_step"), py::arg("sigma_clip"));

    // --- LiveStacker ---
    py::class_<saspro::LiveStacker>(m, "LiveStacker")
        .def(py::init<>())
        .def("reset", &saspro::LiveStacker::reset)
        .def("add_frame", &saspro::LiveStacker::addFrame, py::arg("image"))
        .def("get_mean", &saspro::LiveStacker::getMean)
        .def("get_sigma", &saspro::LiveStacker::getSigma)
        .def("set_sigma_clip", &saspro::LiveStacker::setSigmaClip, py::arg("threshold"))
        .def_property_readonly("frame_count", &saspro::LiveStacker::getFrameCount);
}
