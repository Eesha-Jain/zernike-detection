#ifndef ZERNIKE_EDGE_DETECTION_H
#define ZERNIKE_EDGE_DETECTION_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <complex>
#include <string>
#include <cmath>
#include <limits>

// Forward declarations
// Constants and utility functions are in image_utils.h

// ====================================================================
// COORDINATE TYPE CLASSES
// ====================================================================

/**
 * @brief Polar coordinate in unit circle (l, phi)
 * 
 * Represents a point in polar coordinates on the unit circle:
 * - l: radial distance from center (range: -1 to 1)
 * - phi: angle in radians (range: -π to π)
 */
class PolarCoordinate {
public:
    double l;      // Radial distance from center (unit circle coordinates)
    double phi;    // Angle in radians
    
    PolarCoordinate() : l(0.0), phi(0.0) {}
    PolarCoordinate(double l_val, double phi_val) : l(l_val), phi(phi_val) {}
    
    // Convert to Cartesian coordinates (x, y) given a scale factor
    cv::Point2d toCartesian(double scale = 1.0) const {
        return cv::Point2d(
            scale * l * std::cos(phi),
            scale * l * std::sin(phi)
        );
    }
    
    // Check if coordinate is valid (within unit circle)
    bool isValid() const {
        return std::abs(l) <= 1.0;
    }
};

/**
 * @brief Sub-pixel edge position in image coordinates
 * 
 * Represents an edge point with sub-pixel accuracy:
 * - x: column coordinate (can be fractional)
 * - y: row coordinate (can be fractional)
 */
class EdgePosition {
public:
    double x;  // Column coordinate (sub-pixel)
    double y;  // Row coordinate (sub-pixel)
    
    EdgePosition() : x(0.0), y(0.0) {}
    EdgePosition(double x_val, double y_val) : x(x_val), y(y_val) {}
    EdgePosition(const cv::Point2d& pt) : x(pt.x), y(pt.y) {}
    
    // Convert to OpenCV Point2d
    cv::Point2d toPoint2d() const {
        return cv::Point2d(x, y);
    }
    
    // Convert to integer pixel coordinates (rounded)
    cv::Point toPixel() const {
        return cv::Point(static_cast<int>(std::round(x)), static_cast<int>(std::round(y)));
    }
    
    // Vector operations
    EdgePosition operator+(const EdgePosition& other) const {
        return EdgePosition(x + other.x, y + other.y);
    }
    
    EdgePosition operator-(const EdgePosition& other) const {
        return EdgePosition(x - other.x, y - other.y);
    }
    
    EdgePosition operator*(double scalar) const {
        return EdgePosition(x * scalar, y * scalar);
    }
};

/**
 * @brief Integer pixel coordinate in image
 * 
 * Represents a pixel center position:
 * - x: column index (integer)
 * - y: row index (integer)
 */
class PixelCoordinate {
public:
    int x;  // Column index
    int y;  // Row index
    
    PixelCoordinate() : x(0), y(0) {}
    PixelCoordinate(int x_val, int y_val) : x(x_val), y(y_val) {}
    PixelCoordinate(const cv::Point& pt) : x(pt.x), y(pt.y) {}
    
    // Convert to OpenCV Point
    cv::Point toPoint() const {
        return cv::Point(x, y);
    }
    
    // Convert to EdgePosition (as sub-pixel)
    EdgePosition toEdgePosition() const {
        return EdgePosition(static_cast<double>(x), static_cast<double>(y));
    }
    
    // Convert to OpenCV Point2d
    cv::Point2d toPoint2d() const {
        return cv::Point2d(static_cast<double>(x), static_cast<double>(y));
    }
};

/**
 * @brief Complex Zernike moment (real and imaginary parts)
 * 
 * Represents a Zernike moment as separate real and imaginary components:
 * - real: real part of the moment
 * - imag: imaginary part of the moment
 */
class ZernikeMoment {
public:
    double real;  // Real part
    double imag;  // Imaginary part
    
    ZernikeMoment() : real(0.0), imag(0.0) {}
    ZernikeMoment(double r, double i) : real(r), imag(i) {}
    
    // Get magnitude
    double magnitude() const {
        return std::sqrt(real * real + imag * imag);
    }
    
    // Get phase (angle)
    double phase() const {
        return std::atan2(imag, real);
    }
    
    // Get complex number
    std::complex<double> toComplex() const {
        return std::complex<double>(real, imag);
    }
    
    // Rotate by angle (multiply by exp(-j*phi))
    ZernikeMoment rotate(double phi) const {
        double cos_phi = std::cos(phi);
        double sin_phi = std::sin(phi);
        return ZernikeMoment(
            real * cos_phi + imag * sin_phi,
            -real * sin_phi + imag * cos_phi
        );
    }
    
    // Arithmetic operations
    ZernikeMoment operator+(const ZernikeMoment& other) const {
        return ZernikeMoment(real + other.real, imag + other.imag);
    }
    
    ZernikeMoment operator*(double scalar) const {
        return ZernikeMoment(real * scalar, imag * scalar);
    }
};

/**
 * @brief Edge parameters at a pixel location
 * 
 * Groups all edge detection parameters together:
 * - k: edge strength/intensity
 * - l: distance from pixel center to edge (polar coordinate)
 * - phi: edge angle/orientation (polar coordinate)
 */
class EdgeParameters {
public:
    double k;      // Edge strength parameter
    PolarCoordinate polar;  // Polar coordinates (l, phi)
    
    EdgeParameters() : k(0.0), polar(0.0, 0.0) {}
    EdgeParameters(double k_val, double l_val, double phi_val) 
        : k(k_val), polar(l_val, phi_val) {}
    
    // Check if edge parameters are valid
    bool isValid(double kmin, double kmax, double lmax, double phimin) const {
        return (k > kmin && k < kmax) && 
               (std::abs(polar.l) < lmax) && 
               (std::abs(polar.phi) > phimin);
    }
    
    // Get edge position from pixel center
    EdgePosition getEdgePosition(const PixelCoordinate& pixel_center, int window_size) const {
        cv::Point2d offset = polar.toCartesian(window_size / 2.0);
        EdgePosition pixel_pos = pixel_center.toEdgePosition();
        return EdgePosition(pixel_pos.x + offset.x, pixel_pos.y + offset.y);
    }
};

// ====================================================================
// EDGE DETECTION RESULT STRUCTURE
// ====================================================================

/**
 * @brief Structure to hold edge detection results
 */
struct EdgeResult {
    std::vector<EdgePosition> edges;      // Sub-pixel edge positions
    std::vector<PixelCoordinate> origins; // Original pixel centers
    cv::Mat k;                            // Edge strength map (optional, for debug)
    cv::Mat l;                            // Edge distances map (optional, for debug)
    cv::Mat phi;                          // Edge angles map (optional, for debug)
};

// ====================================================================
// ZERNIKE MOMENTS EDGE DETECTION FUNCTIONS
// ====================================================================
// Based on: Christian (2017) "Accurate Planetary Limb Localization"
// and notes on sub-pixel edge detection using Zernike moments

/**
 * @brief Refine edge positions using Zernike moments sub-pixel detection
 * 
 * This is the main entry point for the Christian-Robinson Zernike moments
 * edge detection algorithm. It takes initial edge points and refines them
 * to sub-pixel accuracy using small windows around each point.
 * 
 * Algorithm:
 * 1. For each initial edge point, extract a small window (mask)
 * 2. Convert window coordinates to unit circle
 * 3. Compute Zernike moments Z_11 and Z_20 for the window
 * 4. Extract edge angle ψ from A_11: ψ = arg(A_11)
 * 5. Solve for edge distance l using equations 61 and 62
 * 6. Convert polar coordinates (l, ψ) back to pixel coordinates
 * 
 * @param image Input grayscale image (CV_64F format)
 * @param initial_edge_points Initial edge points to refine (from Christian-Robinson or other method)
 * @param window_size Size of window around each point (must be odd, typically 7-9)
 * @param transition_width Width of edge transition zone w (default: 1.66 * sigma for Gaussian blur)
 * @param debug If true, returns additional debug information
 * @return EdgeResult containing refined sub-pixel edge positions
 */
EdgeResult refine_edge_positions_with_zernike_moments(
    const cv::Mat& image,
    const std::vector<EdgePosition>& initial_edge_points,
    int window_size = 7,
    double transition_width = 1.66,
    bool debug = false
);

/**
 * @brief Compute Zernike polynomial kernels for a given window size
 * 
 * Creates the Zernike polynomial kernels T_11 and T_20 for a unit circle
 * discretized into a window_size × window_size grid.
 * 
 * Z_11: T_11(r,θ) = r * exp(jθ) = x + jy
 * Z_20: T_20(r,θ) = 2r² - 1 = 2(x²+y²) - 1
 * 
 * @param window_size Size of the window (must be odd)
 * @param kernel_z11_real Output: real part of Z_11 kernel
 * @param kernel_z11_imag Output: imaginary part of Z_11 kernel
 * @param kernel_z20 Output: Z_20 kernel (real only)
 */
void compute_zernike_kernels(
    int window_size,
    cv::Mat& kernel_z11_real,
    cv::Mat& kernel_z11_imag,
    cv::Mat& kernel_z20
);

/**
 * @brief Extract a window around a point in the image
 * 
 * Extracts a window_size × window_size region centered at the given point.
 * Handles boundary conditions by padding with edge values.
 * 
 * @param image Input image
 * @param center_point Center point of the window (sub-pixel coordinates)
 * @param window_size Size of window to extract
 * @return Extracted window as CV_64F matrix
 */
cv::Mat extract_window_around_point(
    const cv::Mat& image,
    const EdgePosition& center_point,
    int window_size
);

/**
 * @brief Compute Zernike moments for a window
 * 
 * Computes the Zernike moments A_11 and A_20 for a given window using
 * discrete summation over pixels within the unit circle.
 * 
 * A_nm = (n+1)/π * Σ Σ I(u,v) * T_nm(u,v)
 * 
 * where the summation is over pixels within the unit circle.
 * 
 * @param window Image window (window_size × window_size)
 * @param kernel_z11_real Real part of Z_11 kernel
 * @param kernel_z11_imag Imaginary part of Z_11 kernel
 * @param kernel_z20 Z_20 kernel
 * @param window_size Size of the window
 * @return ZernikeMoment containing A_11 (real and imag parts)
 * @return A_20 value (output parameter)
 */
ZernikeMoment compute_zernike_moments_for_window(
    const cv::Mat& window,
    const cv::Mat& kernel_z11_real,
    const cv::Mat& kernel_z11_imag,
    const cv::Mat& kernel_z20,
    int window_size,
    double& A20_out
);

/**
 * @brief Extract edge angle from Zernike moment A_11
 * 
 * The edge angle ψ is given by: ψ = arg(A_11) = atan2(Im(A_11), Re(A_11))
 * 
 * This represents the orientation of the edge in the window.
 * 
 * @param A11 Zernike moment A_11
 * @return Edge angle ψ in radians
 */
double extract_edge_angle_from_moment(const ZernikeMoment& A11);

/**
 * @brief Solve for edge distance l using Christian's equations
 * 
 * Solves for the edge distance l from the window center using equations
 * 61 and 62 from Christian (2017), which model the edge as a linear ramp
 * with transition width w.
 * 
 * Equations:
 * A'_11 = (k/(24w)) * {3*asin(l+w) - 3*asin(l-w) - ...}
 * A_20 = (k/(15w)) * [(1-(l-w)²)^(5/2) - (1-(l+w)²)^(5/2)]
 * 
 * This function solves for l given A'_11, A_20, and w.
 * 
 * @param A11_prime Rotated A_11 (real part only, after rotation)
 * @param A20 Zernike moment A_20
 * @param transition_width Width of edge transition zone w
 * @return Edge distance l (in unit circle coordinates, range: -1 to 1)
 */
double solve_edge_distance_from_moments(
    double A11_prime,
    double A20,
    double transition_width
);

/**
 * @brief Convert polar coordinates to pixel coordinates
 * 
 * Converts from unit circle polar coordinates (l, ψ) to pixel coordinates
 * in the image, given the window center and size.
 * 
 * Formula: [u_i; v_i] = [ũ_i; ṽ_i] + (N*l/2) * [cos(ψ); sin(ψ)]
 * 
 * where N is the window size.
 * 
 * @param window_center Center of the window in pixel coordinates
 * @param polar_coord Polar coordinate (l, ψ) in unit circle
 * @param window_size Size of the window
 * @return Refined edge position in pixel coordinates
 */
EdgePosition convert_polar_to_pixel_coordinates(
    const EdgePosition& window_center,
    const PolarCoordinate& polar_coord,
    int window_size
);

// Utility functions
void save_edge_results(const EdgeResult& result, const std::string& filename);
void edge_result_to_opencv(
    const EdgeResult& result,
    std::vector<cv::Point2d>& edges_out,
    std::vector<cv::Point2d>& origins_out
);

#endif // ZERNIKE_EDGE_DETECTION_H
