/**
 * @file example_helper_usage.cpp
 * @brief Example demonstrating the usage of helper functions for JSON parsing and visualization
 * 
 * This example shows how to:
 * 1. Read edge points from a JSON file
 * 2. Display edge points on an image
 * 3. Use the coordinate type classes
 */

#include "helper.h"
#include "image_utils.h"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <json_file> <image_file>" << std::endl;
        std::cout << "Example: " << argv[0] << " chosen_points.json moon.jpg" << std::endl;
        return 1;
    }
    
    std::string json_file = argv[1];
    std::string image_file = argv[2];
    
    std::cout << "=== Helper Functions Example ===" << std::endl;
    
    // 1. Read edge points from JSON file
    std::cout << "\n1. Reading edge points from JSON file: " << json_file << std::endl;
    std::vector<EdgePosition> edge_points = read_edge_points_from_json(json_file);
    
    if (edge_points.empty()) {
        std::cerr << "Error: No edge points loaded from JSON file" << std::endl;
        return 1;
    }
    
    std::cout << "   Loaded " << edge_points.size() << " edge points" << std::endl;
    std::cout << "   First point: (" << edge_points[0].x << ", " << edge_points[0].y << ")" << std::endl;
    
    // 2. Load image
    std::cout << "\n2. Loading image: " << image_file << std::endl;
    cv::Mat image = cv::imread(image_file, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Could not load image " << image_file << std::endl;
        return 1;
    }
    std::cout << "   Image size: " << image.cols << "x" << image.rows << std::endl;
    
    // 3. Display edge points on image
    std::cout << "\n3. Displaying edge points on image..." << std::endl;
    cv::Mat result = display_edge_points_on_image(
        image,
        edge_points,
        cv::Scalar(0, 255, 0),  // Green color
        5,                       // Point radius
        true,                    // Show image
        "Edge Points from JSON"  // Window name
    );
    
    // 4. Save the result
    std::string output_file = "edge_points_visualization.jpg";
    cv::imwrite(output_file, result);
    std::cout << "\n4. Saved visualization to: " << output_file << std::endl;
    
    std::cout << "\n=== Example Complete ===" << std::endl;
    return 0;
}

