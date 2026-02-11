#include "zernike_edge_detection.h"
#include "image_utils.h"
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <fstream>

// Use SMALL from image_utils
extern const double SMALL;

// ====================================================================
// ZERNIKE KERNEL COMPUTATION
// ====================================================================

void compute_zernike_kernels(
    int window_size,
    cv::Mat& kernel_z11_real,
    cv::Mat& kernel_z11_imag,
    cv::Mat& kernel_z20
) {
    // Ensure window_size is odd
    if (window_size % 2 != 1) {
        std::cerr << "Warning: window_size must be odd. Adjusting to " << (window_size + 1) << std::endl;
        window_size = window_size + 1;
    }
    
    // Initialize kernels
    kernel_z11_real = cv::Mat::zeros(window_size, window_size, CV_64F);
    kernel_z11_imag = cv::Mat::zeros(window_size, window_size, CV_64F);
    kernel_z20 = cv::Mat::zeros(window_size, window_size, CV_64F);
    
    // Offset to center kernel around (0,0)
    // For window_size = 7: offset = 1 - 1/7 = 6/7
    double offset = 1.0 * (1.0 - 1.0 / window_size);
    
    // Normalization factor: (n+1)/π
    double normalization_11 = (1.0 + 1.0) / CV_PI;  // n=1
    double normalization_20 = (2.0 + 1.0) / CV_PI;  // n=2
    
    // Fill kernels
    for (int i = 0; i < window_size; i++) {
        for (int j = 0; j < window_size; j++) {
            // Convert pixel coordinates to unit circle coordinates [-1, 1]
            double u = 2.0 * j / window_size - offset;  // x-coordinate
            double v = 2.0 * i / window_size - offset;  // y-coordinate
            
            // Only compute within unit circle: u² + v² ≤ 1
            double r_squared = u * u + v * v;
            if (r_squared <= 1.0) {
                // Z_11 polynomial: T_11(r,θ) = r * exp(jθ) = u + jv
                // Real part: u, Imaginary part: v
                kernel_z11_real.at<double>(i, j) = u * normalization_11;
                kernel_z11_imag.at<double>(i, j) = v * normalization_11;
                
                // Z_20 polynomial: T_20(r,θ) = 2r² - 1 = 2(u²+v²) - 1
                kernel_z20.at<double>(i, j) = (2.0 * r_squared - 1.0) * normalization_20;
            }
        }
    }
}

// ====================================================================
// WINDOW EXTRACTION
// ====================================================================

cv::Mat extract_window_around_point(
    const cv::Mat& image,
    const EdgePosition& center_point,
    int window_size
) {
    int half_window = window_size / 2;
    cv::Mat window = cv::Mat::zeros(window_size, window_size, CV_64F);
    
    // Get integer pixel coordinates
    int center_x = static_cast<int>(std::round(center_point.x));
    int center_y = static_cast<int>(std::round(center_point.y));
    
    // Extract window with boundary handling
    for (int i = 0; i < window_size; i++) {
        for (int j = 0; j < window_size; j++) {
            // Calculate pixel coordinates in image
            int img_x = center_x + (j - half_window);
            int img_y = center_y + (i - half_window);
            
            // Handle boundaries: use edge values
            img_x = std::max(0, std::min(img_x, image.cols - 1));
            img_y = std::max(0, std::min(img_y, image.rows - 1));
            
            window.at<double>(i, j) = image.at<double>(img_y, img_x);
        }
    }
    
    return window;
}

// ====================================================================
// ZERNIKE MOMENTS COMPUTATION
// ====================================================================

ZernikeMoment compute_zernike_moments_for_window(
    const cv::Mat& window,
    const cv::Mat& kernel_z11_real,
    const cv::Mat& kernel_z11_imag,
    const cv::Mat& kernel_z20,
    int window_size,
    double& A20_out
) {
    // Initialize moments
    double A11_real = 0.0;
    double A11_imag = 0.0;
    double A20 = 0.0;
    
    // Compute moments by summing over all pixels in the window
    // A_nm = Σ Σ I(u,v) * T_nm(u,v)
    for (int i = 0; i < window_size; i++) {
        for (int j = 0; j < window_size; j++) {
            double intensity = window.at<double>(i, j);
            
            // Only sum if kernel value is non-zero (within unit circle)
            double k11_real = kernel_z11_real.at<double>(i, j);
            double k11_imag = kernel_z11_imag.at<double>(i, j);
            double k20 = kernel_z20.at<double>(i, j);
            
            // Accumulate moments
            A11_real += intensity * k11_real;
            A11_imag += intensity * k11_imag;
            A20 += intensity * k20;
        }
    }
    
    A20_out = A20;
    return ZernikeMoment(A11_real, A11_imag);
}

// ====================================================================
// EDGE ANGLE EXTRACTION
// ====================================================================

double extract_edge_angle_from_moment(const ZernikeMoment& A11) {
    // Edge angle: ψ = arg(A_11) = atan2(Im(A_11), Re(A_11))
    return std::atan2(A11.imag, A11.real);
}

// ====================================================================
// EDGE DISTANCE SOLVING (Christian's Equations 61 & 62)
// ====================================================================

double solve_edge_distance_from_moments(
    double A11_prime,
    double A20,
    double transition_width
) {
    // Equations from Christian (2017), Equations 61 & 62:
    // A'_11 = (k/(24w)) * {3*asin(l+w) - 3*asin(l-w) - 
    //          (5(l-w) - 2(l-w)³)√(1-(l-w)²) + 
    //          (5(l+w) - 2(l+w)³)√(1-(l+w)²)}
    // A_20 = (k/(15w)) * [(1-(l-w)²)^(5/2) - (1-(l+w)²)^(5/2)]
    // 
    // We solve for l using a binary search approach, since the relationship is non-linear.
    
    double w = transition_width;
    
    // Handle edge cases
    if (std::abs(A20) < SMALL && std::abs(A11_prime) < SMALL) {
        return 0.0;  // Edge at center
    }
    
    // Helper function to compute A'_11 and A_20 for a given l
    auto compute_moments_for_l = [w](double l) -> std::pair<double, double> {
        double l_plus_w = std::max(-1.0, std::min(1.0, l + w));
        double l_minus_w = std::max(-1.0, std::min(1.0, l - w));
        
        // Compute A_20 (Equation 62)
        double term1 = std::pow(std::max(0.0, 1.0 - l_minus_w * l_minus_w), 2.5);
        double term2 = std::pow(std::max(0.0, 1.0 - l_plus_w * l_plus_w), 2.5);
        double A20_val = (term1 - term2) / (15.0 * w);
        
        // Compute A'_11 (Equation 61) - we'll compute this normalized by k
        double asin_lpw = std::asin(l_plus_w);
        double asin_lmw = std::asin(l_minus_w);
        
        double sqrt1_lmw = std::sqrt(std::max(0.0, 1.0 - l_minus_w * l_minus_w));
        double sqrt1_lpw = std::sqrt(std::max(0.0, 1.0 - l_plus_w * l_plus_w));
        
        double term3 = (5.0 * l_minus_w - 2.0 * std::pow(l_minus_w, 3.0)) * sqrt1_lmw;
        double term4 = (5.0 * l_plus_w - 2.0 * std::pow(l_plus_w, 3.0)) * sqrt1_lpw;
        
        double A11_val = (3.0 * asin_lpw - 3.0 * asin_lmw - term3 + term4) / (24.0 * w);
        
        return std::make_pair(A11_val, A20_val);
    };
    
    // Use binary search to find l that minimizes the error
    double l_min = -1.0 + SMALL;
    double l_max = 1.0 - SMALL;
    double best_l = 0.0;
    double best_error = std::numeric_limits<double>::max();
    
    // Binary search with fine resolution
    int num_iterations = 50;
    for (int iter = 0; iter < num_iterations; iter++) {
        double l_test = (l_min + l_max) / 2.0;
        
        auto [A11_expected, A20_expected] = compute_moments_for_l(l_test);
        
        // If both are near zero, skip
        if (std::abs(A20_expected) < SMALL) {
            l_test = (l_test > 0) ? l_test - 0.01 : l_test + 0.01;
            if (l_test < l_min || l_test > l_max) continue;
            auto [A11_alt, A20_alt] = compute_moments_for_l(l_test);
            A11_expected = A11_alt;
            A20_expected = A20_alt;
        }
        
        if (std::abs(A20_expected) < SMALL) continue;
        
        // Estimate k from A_20
        double k_est = A20 / A20_expected;
        double A11_expected_scaled = A11_expected * k_est;
        
        // Compute error
        double error = std::abs(A11_prime - A11_expected_scaled);
        
        if (error < best_error) {
            best_error = error;
            best_l = l_test;
        }
        
        // Update search bounds
        if (A11_expected_scaled < A11_prime) {
            l_min = l_test;
        } else {
            l_max = l_test;
        }
    }
    
    // Final refinement around best_l
    double refine_step = 0.001;
    for (double l_test = best_l - 0.01; l_test <= best_l + 0.01; l_test += refine_step) {
        if (l_test < -1.0 + SMALL || l_test > 1.0 - SMALL) continue;
        
        auto [A11_expected, A20_expected] = compute_moments_for_l(l_test);
        if (std::abs(A20_expected) < SMALL) continue;
        
        double k_est = A20 / A20_expected;
        double A11_expected_scaled = A11_expected * k_est;
        double error = std::abs(A11_prime - A11_expected_scaled);
        
        if (error < best_error) {
            best_error = error;
            best_l = l_test;
        }
    }
    
    return best_l;
}

// ====================================================================
// COORDINATE CONVERSION
// ====================================================================

EdgePosition convert_polar_to_pixel_coordinates(
    const EdgePosition& window_center,
    const PolarCoordinate& polar_coord,
    int window_size
) {
    // Formula: [u_i; v_i] = [ũ_i; ṽ_i] + (N*l/2) * [cos(ψ); sin(ψ)]
    // where N is window_size, l is distance, ψ is angle
    
    double scale = window_size / 2.0;
    cv::Point2d offset = polar_coord.toCartesian(scale);
    
    return EdgePosition(
        window_center.x + offset.x,
        window_center.y + offset.y
    );
}

// ====================================================================
// MAIN ALGORITHM
// ====================================================================

EdgeResult refine_edge_positions_with_zernike_moments(
    const cv::Mat& image,
    const std::vector<EdgePosition>& initial_edge_points,
    int window_size,
    double transition_width,
    bool debug
) {
    EdgeResult result;
    
    // Validate inputs
    if (image.empty()) {
        std::cerr << "Error: Input image is empty" << std::endl;
        return result;
    }
    
    if (initial_edge_points.empty()) {
        std::cerr << "Error: No initial edge points provided" << std::endl;
        return result;
    }
    
    // Ensure window_size is odd
    if (window_size % 2 != 1) {
        window_size = window_size + 1;
        std::cout << "Adjusted window_size to " << window_size << " (must be odd)" << std::endl;
    }
    
    // Step 1: Compute Zernike kernels once (reused for all windows)
    cv::Mat kernel_z11_real, kernel_z11_imag, kernel_z20;
    compute_zernike_kernels(window_size, kernel_z11_real, kernel_z11_imag, kernel_z20);
    
    // Step 2: Process each initial edge point
    std::vector<EdgePosition> refined_edges;
    std::vector<PixelCoordinate> origins;
    
    for (const auto& initial_point : initial_edge_points) {
        // Step 2a: Extract window around the point
        cv::Mat window = extract_window_around_point(image, initial_point, window_size);
        
        // Step 2b: Compute Zernike moments for this window
        double A20 = 0.0;
        ZernikeMoment A11 = compute_zernike_moments_for_window(
            window, kernel_z11_real, kernel_z11_imag, kernel_z20, window_size, A20
        );
        
        // Step 2c: Extract edge angle ψ from A_11
        double psi = extract_edge_angle_from_moment(A11);
        
        // Step 2d: Rotate A_11 to align with edge direction
        // A'_11 = Re(A_11)*cos(ψ) + Im(A_11)*sin(ψ)
        double cos_psi = std::cos(psi);
        double sin_psi = std::sin(psi);
        double A11_prime = A11.real * cos_psi + A11.imag * sin_psi;
        
        // Step 2e: Solve for edge distance l using Christian's equations
        double l = solve_edge_distance_from_moments(A11_prime, A20, transition_width);
        
        // Step 2f: Create polar coordinate
        PolarCoordinate polar_coord(l, psi);
        
        // Step 2g: Convert back to pixel coordinates
        EdgePosition refined_edge = convert_polar_to_pixel_coordinates(
            initial_point, polar_coord, window_size
        );
        
        // Store results
        refined_edges.push_back(refined_edge);
        
        // Store origin (rounded initial point)
        PixelCoordinate origin(
            static_cast<int>(std::round(initial_point.x)),
            static_cast<int>(std::round(initial_point.y))
        );
        origins.push_back(origin);
    }
    
    result.edges = refined_edges;
    result.origins = origins;
    
    return result;
}

// ====================================================================
// UTILITY FUNCTIONS
// ====================================================================

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
