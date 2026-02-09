#ifndef HELPER_H
#define HELPER_H

#include "zernike_edge_detection.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

/**
 * @brief Read edge points from a JSON file
 * 
 * Reads a JSON file in the format:
 * {
 *   "points": [[x1, y1], [x2, y2], ...],
 *   "count": N
 * }
 * 
 * @param filename Path to JSON file
 * @return Vector of EdgePosition objects representing the edge points
 */
std::vector<EdgePosition> read_edge_points_from_json(const std::string& filename);

/**
 * @brief Display edge points on an image
 * 
 * Draws edge points on the given image with customizable visualization options.
 * 
 * @param image Input image (will be modified/displayed)
 * @param edge_points Vector of edge positions to display
 * @param point_color Color for edge points (BGR format, default: blue)
 * @param point_radius Radius of points in pixels (default: 3)
 * @param show_image If true, displays the image using cv::imshow (default: true)
 * @param window_name Window name for display (default: "Edge Points")
 * @return Image with edge points drawn on it
 */
cv::Mat display_edge_points_on_image(
    const cv::Mat& image,
    const std::vector<EdgePosition>& edge_points,
    const cv::Scalar& point_color = cv::Scalar(255, 0, 0),  // Blue in BGR
    int point_radius = 3,
    bool show_image = true,
    const std::string& window_name = "Edge Points"
);

/**
 * @brief Display edge points with optional origin vectors
 * 
 * Similar to display_edge_points_on_image but can also show vectors
 * from pixel centers to edge positions (like the Python version).
 * 
 * @param image Input image
 * @param edge_points Vector of edge positions
 * @param origin_points Vector of pixel origins (optional, can be empty)
 * @param point_color Color for edge points (default: blue)
 * @param vector_color Color for vectors (default: orange)
 * @param point_radius Radius of points (default: 3)
 * @param show_image If true, displays the image (default: true)
 * @param window_name Window name (default: "Edge Detection")
 * @return Image with visualization
 */
cv::Mat display_edge_detection_result(
    const cv::Mat& image,
    const std::vector<EdgePosition>& edge_points,
    const std::vector<PixelCoordinate>& origin_points = {},
    const cv::Scalar& point_color = cv::Scalar(255, 0, 0),  // Blue
    const cv::Scalar& vector_color = cv::Scalar(0, 165, 255),  // Orange
    int point_radius = 3,
    bool show_image = true,
    const std::string& window_name = "Edge Detection"
);

#endif // HELPER_H

