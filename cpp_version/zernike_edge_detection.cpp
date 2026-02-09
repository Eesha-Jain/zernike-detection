#include "zernike_edge_detection.h"
#include "image_utils.h"
#include <iostream>
#include <algorithm>
#include <iomanip>

// Use SMALL from image_utils
extern const double SMALL;


// Main Zernike edge detection function
EdgeResult ghosal_edge_v2(
    const cv::Mat& img,
    int Ks,
    double kmin,
    double kmax,
    double lmax,
    double phimin,
    bool thresholding,
    bool debug,
    bool mirror
) {
    EdgeResult result;
    
    // Gather image properties before it's altered
    int ni = img.rows;
    int nj = img.cols;
    
    // Ks must be odd
    if (Ks % 2 != 1) {
        std::cout << "Ks must be odd! Continuing with Ks = Ks-1" << std::endl;
        Ks = Ks - 1;
    }
    
    // ====================================================================
    // STEP 1: CONSTRUCT ZERNIKE POLYNOMIAL KERNELS
    // ====================================================================
    cv::Mat Vc11_real = cv::Mat::zeros(Ks, Ks, CV_64F);
    cv::Mat Vc11_imag = cv::Mat::zeros(Ks, Ks, CV_64F);
    cv::Mat Vc20 = cv::Mat::zeros(Ks, Ks, CV_64F);
    
    double ofs = 1.0 * (1.0 - 1.0 / Ks);  // offset for centering kernel around (0,0)
    
    for (int i = 0; i < Ks; i++) {
        for (int j = 0; j < Ks; j++) {
            // Normalize pixel coordinates to unit disk [-1, 1]
            double Kx = 2.0 * j / Ks - ofs;  // x-coordinate in unit circle
            double Ky = 2.0 * i / Ks - ofs;  // y-coordinate in unit circle
            
            // Only compute within unit circle
            if (Kx * Kx + Ky * Ky <= 1.0) {
                // Z₁₁ polynomial: T₁₁(r,θ) = r * exp(jθ) = (x + jy) = Kx - j*Ky
                Vc11_real.at<double>(i, j) = Kx;
                Vc11_imag.at<double>(i, j) = -Ky;
                
                // Z₂₀ polynomial: T₂₀(r,θ) = 2r² - 1 = 2(x²+y²) - 1
                Vc20.at<double>(i, j) = 2.0 * Kx * Kx + 2.0 * Ky * Ky - 1.0;
            }
        }
    }
    
    // Mirror image edges to avoid convolution artifacts at boundaries
    cv::Mat img_processed = img.clone();
    int border_type = mirror ? cv::BORDER_REFLECT_101 : cv::BORDER_CONSTANT;
    
    // ====================================================================
    // STEP 2: COMPUTE ZERNIKE MOMENTS
    // ====================================================================
    double Anorm_1 = (1.0 + 1.0) / CV_PI;  // Normalization factor for n=1
    double Anorm_2 = (2.0 + 1.0) / CV_PI;  // Normalization factor for n=2
    
    // Compute Zernike moments via convolution
    cv::Mat Vc11_real_norm, Vc11_imag_norm, Vc20_norm;
    Vc11_real.convertTo(Vc11_real_norm, CV_64F);
    Vc11_imag.convertTo(Vc11_imag_norm, CV_64F);
    Vc20.convertTo(Vc20_norm, CV_64F);
    
    Vc11_real_norm *= Anorm_1;
    Vc11_imag_norm *= Anorm_1;
    Vc20_norm *= Anorm_2;
    
    cv::Mat A11_real, A11_imag;
    cv::filter2D(img_processed, A11_real, CV_64F, Vc11_real_norm, cv::Point(-1, -1), 0, border_type);
    cv::filter2D(img_processed, A11_imag, CV_64F, Vc11_imag_norm, cv::Point(-1, -1), 0, border_type);
    
    cv::Mat A20;
    cv::filter2D(img_processed, A20, CV_64F, Vc20_norm, cv::Point(-1, -1), 0, border_type);
    
    // ====================================================================
    // STEP 3: EXTRACT EDGE PARAMETERS FROM ZERNIKE MOMENTS
    // ====================================================================
    
    // Calculate edge angle φ
    cv::Mat A11_real_safe = zero_to_small(A11_real);
    cv::Mat phi_atan = cv::Mat::zeros(ni, nj, CV_64F);
    cv::phase(A11_real, A11_imag, phi_atan);  // More accurate than atan
    
    // Rotate A₁₁ to align with edge direction: A'₁₁ = Re(A₁₁)*cos(φ) + Im(A₁₁)*sin(φ)
    cv::Mat cos_phi, sin_phi;
    cv::cos(phi_atan, cos_phi);
    cv::sin(phi_atan, sin_phi);
    cv::Mat Al11 = A11_real.mul(cos_phi) + A11_imag.mul(sin_phi);
    
    // Calculate edge distance l from pixel center
    cv::Mat Al11_safe = zero_to_small(Al11);
    cv::Mat l;
    cv::divide(A20, Al11_safe, l);
    
    // Clamp l to valid range [-1, 1]
    cv::Mat l_clamped;
    extern const double SMALL;  // Defined in image_utils.cpp
    cv::max(l, -1.0 + SMALL, l_clamped);
    cv::min(l_clamped, 1.0 - SMALL, l);
    
    // Calculate edge strength parameter k
    cv::Mat l_squared;
    cv::multiply(l, l, l_squared);
    cv::Mat one_minus_l_squared = 1.0 - l_squared;
    cv::Mat denominator;
    cv::pow(one_minus_l_squared, 1.5, denominator);
    cv::Mat k = cv::abs(3.0 * Al11 / (2.0 * denominator));
    
    // ====================================================================
    // STEP 4: FILTER EDGE DETECTIONS BY QUALITY CRITERIA
    // ====================================================================
    cv::Mat valid = cv::Mat::ones(ni, nj, CV_8U);
    
    if (thresholding) {
        // Create boolean masks for each quality criterion
        cv::Mat phi_c, l_c, k_c;
        cv::compare(cv::abs(phi_atan), phimin, phi_c, cv::CMP_GT);
        cv::compare(cv::abs(l), lmax, l_c, cv::CMP_LT);
        cv::Mat k_min_mask, k_max_mask;
        cv::compare(k, kmin, k_min_mask, cv::CMP_GT);
        cv::compare(k, kmax, k_max_mask, cv::CMP_LT);
        cv::bitwise_and(k_min_mask, k_max_mask, k_c);
        
        // Combine all conditions
        cv::bitwise_and(phi_c, l_c, valid);
        cv::bitwise_and(valid, k_c, valid);
    }
    
    // Extract valid edge points
    std::vector<EdgePosition> edges;
    std::vector<PixelCoordinate> origins;
    
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++) {
            if (valid.at<uchar>(i, j)) {
                double l_val = l.at<double>(i, j);
                double phi_val = phi_atan.at<double>(i, j);
                
                // Create polar coordinate
                PolarCoordinate polar(l_val, phi_val);
                
                // Create pixel coordinate for origin
                PixelCoordinate pixel_origin(j, i);  // (x, y) = (column, row)
                
                // Convert polar coordinates to edge position
                cv::Point2d offset = polar.toCartesian(Ks / 2.0);
                EdgePosition edge_pos(
                    pixel_origin.x + offset.x,
                    pixel_origin.y + offset.y
                );
                
                edges.push_back(edge_pos);
                origins.push_back(pixel_origin);
            }
        }
    }
    
    result.edges = edges;
    result.origins = origins;
    
    if (debug) {
        result.k = k;
        result.l = l;
        result.phi = phi_atan;
    }
    
    return result;
}

// Save edge results to file
void save_edge_results(const EdgeResult& result, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing " << filename << std::endl;
        return;
    }
    
    file << std::fixed << std::setprecision(6);
    for (const auto& edge : result.edges) {
        file << edge.x << "\t" << edge.y << "\n";
    }
    
    file.close();
}

// Convert EdgeResult to OpenCV format for visualization
void edge_result_to_opencv(
    const EdgeResult& result,
    std::vector<cv::Point2d>& edges_out,
    std::vector<cv::Point2d>& origins_out
) {
    edges_out.clear();
    origins_out.clear();
    
    for (const auto& edge : result.edges) {
        edges_out.push_back(edge.toPoint2d());
    }
    
    for (const auto& origin : result.origins) {
        origins_out.push_back(origin.toPoint2d());
    }
}

