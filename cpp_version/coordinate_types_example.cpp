/**
 * @file coordinate_types_example.cpp
 * @brief Example demonstrating the usage of coordinate type classes
 * 
 * This file shows how to use the different coordinate types:
 * - PolarCoordinate: for polar coordinates (l, phi)
 * - EdgePosition: for sub-pixel edge positions
 * - PixelCoordinate: for integer pixel coordinates
 * - ZernikeMoment: for complex Zernike moments
 * - EdgeParameters: for grouped edge parameters
 */

#include "zernike_edge_detection.h"
#include <iostream>

void demonstrate_coordinate_types() {
    std::cout << "=== Coordinate Types Demonstration ===\n\n";
    
    // 1. PolarCoordinate example
    std::cout << "1. PolarCoordinate:\n";
    PolarCoordinate polar(0.5, 1.5708);  // l=0.5, phi=π/2
    std::cout << "   Created: l=" << polar.l << ", phi=" << polar.phi << " rad\n";
    cv::Point2d cart = polar.toCartesian(10.0);  // Scale by 10
    std::cout << "   Converted to Cartesian (scale=10): (" 
              << cart.x << ", " << cart.y << ")\n";
    std::cout << "   Valid: " << (polar.isValid() ? "Yes" : "No") << "\n\n";
    
    // 2. EdgePosition example
    std::cout << "2. EdgePosition:\n";
    EdgePosition edge1(123.456, 789.012);
    std::cout << "   Created: (" << edge1.x << ", " << edge1.y << ")\n";
    cv::Point pixel = edge1.toPixel();
    std::cout << "   Rounded to pixel: (" << pixel.x << ", " << pixel.y << ")\n";
    
    EdgePosition edge2(100.0, 200.0);
    EdgePosition edge_diff = edge1 - edge2;
    std::cout << "   Difference: (" << edge_diff.x << ", " << edge_diff.y << ")\n\n";
    
    // 3. PixelCoordinate example
    std::cout << "3. PixelCoordinate:\n";
    PixelCoordinate pixel_coord(42, 84);
    std::cout << "   Created: (" << pixel_coord.x << ", " << pixel_coord.y << ")\n";
    EdgePosition edge_from_pixel = pixel_coord.toEdgePosition();
    std::cout << "   Converted to EdgePosition: (" 
              << edge_from_pixel.x << ", " << edge_from_pixel.y << ")\n\n";
    
    // 4. ZernikeMoment example
    std::cout << "4. ZernikeMoment:\n";
    ZernikeMoment moment(3.0, 4.0);  // 3 + 4j
    std::cout << "   Created: " << moment.real << " + " << moment.imag << "j\n";
    std::cout << "   Magnitude: " << moment.magnitude() << "\n";
    std::cout << "   Phase: " << moment.phase() << " rad\n";
    
    ZernikeMoment rotated = moment.rotate(0.7854);  // Rotate by π/4
    std::cout << "   Rotated by π/4: " << rotated.real << " + " 
              << rotated.imag << "j\n\n";
    
    // 5. EdgeParameters example
    std::cout << "5. EdgeParameters:\n";
    EdgeParameters params(50.0, 0.3, 1.2);  // k=50, l=0.3, phi=1.2
    std::cout << "   Created: k=" << params.k 
              << ", l=" << params.polar.l 
              << ", phi=" << params.polar.phi << "\n";
    
    PixelCoordinate center(100, 200);
    EdgePosition edge_pos = params.getEdgePosition(center, 15);  // Ks=15
    std::cout << "   Edge position from pixel center (100, 200): (" 
              << edge_pos.x << ", " << edge_pos.y << ")\n";
    
    bool valid = params.isValid(10.0, 100.0, 0.5, 1.0);
    std::cout << "   Valid (kmin=10, kmax=100, lmax=0.5, phimin=1.0): " 
              << (valid ? "Yes" : "No") << "\n\n";
    
    std::cout << "=== End Demonstration ===\n";
}

// Uncomment to run demonstration
// int main() {
//     demonstrate_coordinate_types();
//     return 0;
// }

