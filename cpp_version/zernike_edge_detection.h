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
    EdgePosition getEdgePosition(const PixelCoordinate& pixel_center, int Ks) const {
        cv::Point2d offset = polar.toCartesian(Ks / 2.0);
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
    std::vector<PixelCoordinate> origins;   // Original pixel centers
    cv::Mat k;                            // Edge strength map (optional, for debug)
    cv::Mat l;                            // Edge distances map (optional, for debug)
    cv::Mat phi;                          // Edge angles map (optional, for debug)
};

// Note: Image I/O and utility functions are in image_utils.h and file_utils.h

// Main edge detection function
EdgeResult ghosal_edge_v2(
    const cv::Mat& img,
    int Ks,
    double kmin = 0.0,
    double kmax = 1000.0,
    double lmax = 0.5,
    double phimin = 1.0,
    bool thresholding = true,
    bool debug = false,
    bool mirror = false
);

// Utility function to save edge results
void save_edge_results(const EdgeResult& result, const std::string& filename);

// Utility function to convert EdgeResult to OpenCV format (for visualization)
void edge_result_to_opencv(
    const EdgeResult& result,
    std::vector<cv::Point2d>& edges_out,
    std::vector<cv::Point2d>& origins_out
);

#endif // ZERNIKE_EDGE_DETECTION_H

