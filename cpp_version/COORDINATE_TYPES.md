# Coordinate Types Documentation

This document describes the coordinate type classes used in the Zernike edge detection implementation.

## Overview

The codebase uses strongly-typed coordinate classes to improve type safety, prevent coordinate system confusion, and make the code more self-documenting. Each coordinate type has a specific purpose and coordinate system.

## Class Hierarchy

```
PolarCoordinate          - Unit circle polar coordinates (l, phi)
EdgePosition            - Sub-pixel edge positions (x, y)
PixelCoordinate         - Integer pixel coordinates (x, y)
ZernikeMoment           - Complex Zernike moments (real, imag)
EdgeParameters          - Grouped edge parameters (k, polar)
```

## Detailed Class Descriptions

### `PolarCoordinate`

Represents a point in polar coordinates on the unit circle, used internally for Zernike moment calculations.

**Members:**
- `double l`: Radial distance from center (range: -1.0 to 1.0)
- `double phi`: Angle in radians (range: -π to π)

**Methods:**
- `PolarCoordinate(double l, double phi)`: Constructor
- `cv::Point2d toCartesian(double scale = 1.0)`: Convert to Cartesian coordinates
- `bool isValid()`: Check if coordinate is within unit circle

**Usage:**
```cpp
PolarCoordinate polar(0.5, 1.5708);  // l=0.5, phi=π/2
cv::Point2d cart = polar.toCartesian(10.0);  // Scale by 10
```

### `EdgePosition`

Represents a sub-pixel edge position in image coordinates. Used for the final output edge points.

**Members:**
- `double x`: Column coordinate (can be fractional)
- `double y`: Row coordinate (can be fractional)

**Methods:**
- `EdgePosition(double x, double y)`: Constructor
- `cv::Point2d toPoint2d()`: Convert to OpenCV Point2d
- `cv::Point toPixel()`: Round to nearest integer pixel
- Arithmetic operators: `+`, `-`, `*` (scalar)

**Usage:**
```cpp
EdgePosition edge(123.456, 789.012);
cv::Point2d pt = edge.toPoint2d();
cv::Point pixel = edge.toPixel();  // (123, 789)
```

### `PixelCoordinate`

Represents an integer pixel coordinate (pixel center position).

**Members:**
- `int x`: Column index
- `int y`: Row index

**Methods:**
- `PixelCoordinate(int x, int y)`: Constructor
- `cv::Point toPoint()`: Convert to OpenCV Point
- `EdgePosition toEdgePosition()`: Convert to EdgePosition (as sub-pixel)
- `cv::Point2d toPoint2d()`: Convert to OpenCV Point2d

**Usage:**
```cpp
PixelCoordinate pixel(42, 84);
EdgePosition edge = pixel.toEdgePosition();  // (42.0, 84.0)
```

### `ZernikeMoment`

Represents a complex Zernike moment with separate real and imaginary parts.

**Members:**
- `double real`: Real part of the moment
- `double imag`: Imaginary part of the moment

**Methods:**
- `ZernikeMoment(double r, double i)`: Constructor
- `double magnitude()`: Compute magnitude |z|
- `double phase()`: Compute phase angle arg(z)
- `std::complex<double> toComplex()`: Convert to std::complex
- `ZernikeMoment rotate(double phi)`: Rotate by angle phi
- Arithmetic operators: `+`, `*` (scalar)

**Usage:**
```cpp
ZernikeMoment moment(3.0, 4.0);  // 3 + 4j
double mag = moment.magnitude();  // 5.0
double phase = moment.phase();     // atan2(4, 3)
```

### `EdgeParameters`

Groups all edge detection parameters together for a single pixel location.

**Members:**
- `double k`: Edge strength parameter
- `PolarCoordinate polar`: Polar coordinates (l, phi)

**Methods:**
- `EdgeParameters(double k, double l, double phi)`: Constructor
- `bool isValid(double kmin, double kmax, double lmax, double phimin)`: Check validity
- `EdgePosition getEdgePosition(const PixelCoordinate& center, int Ks)`: Compute edge position

**Usage:**
```cpp
EdgeParameters params(50.0, 0.3, 1.2);  // k=50, l=0.3, phi=1.2
PixelCoordinate center(100, 200);
EdgePosition edge = params.getEdgePosition(center, 15);  // Ks=15
```

## EdgeResult Structure

The `EdgeResult` structure contains the output of edge detection:

```cpp
struct EdgeResult {
    std::vector<EdgePosition> edges;      // Sub-pixel edge positions
    std::vector<PixelCoordinate> origins; // Original pixel centers
    cv::Mat k;                            // Edge strength map (debug)
    cv::Mat l;                            // Edge distances map (debug)
    cv::Mat phi;                          // Edge angles map (debug)
};
```

## Coordinate System Conventions

1. **Image Coordinates**: 
   - Origin (0,0) at top-left corner
   - x-axis: increases rightward (columns)
   - y-axis: increases downward (rows)

2. **Polar Coordinates**:
   - l: distance from center (unit circle: -1 to 1)
   - phi: angle measured from positive x-axis (counter-clockwise)

3. **Sub-pixel Precision**:
   - Edge positions can have fractional pixel coordinates
   - Pixel coordinates are always integers

## Benefits of Type Safety

1. **Prevents Errors**: Compiler catches coordinate system mismatches
2. **Self-Documenting**: Code clearly shows what coordinate system is being used
3. **Easier Debugging**: Type names indicate coordinate system in debugger
4. **Better IDE Support**: Autocomplete shows relevant methods for each type

## Example: Complete Edge Detection Workflow

```cpp
// 1. Detect edges
EdgeResult result = ghosal_edge_v2(img, Ks, kmin, kmax, lmax, phimin);

// 2. Access edge positions (sub-pixel)
for (const auto& edge : result.edges) {
    std::cout << "Edge at: (" << edge.x << ", " << edge.y << ")\n";
}

// 3. Access pixel origins
for (const auto& origin : result.origins) {
    std::cout << "Origin pixel: (" << origin.x << ", " << origin.y << ")\n";
}

// 4. Convert to OpenCV format for visualization
std::vector<cv::Point2d> edges_cv, origins_cv;
edge_result_to_opencv(result, edges_cv, origins_cv);
```

