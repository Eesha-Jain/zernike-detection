#include "helper.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>

// Simple JSON parser for edge points
// Note: This is a basic parser. For production, consider using a proper JSON library like nlohmann/json
std::vector<EdgePosition> read_edge_points_from_json(const std::string& filename) {
    std::vector<EdgePosition> edge_points;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open JSON file " << filename << std::endl;
        return edge_points;
    }
    
    // Read entire file into string
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json_content = buffer.str();
    file.close();
    
    // Find the "points" array
    size_t points_start = json_content.find("\"points\"");
    if (points_start == std::string::npos) {
        std::cerr << "Error: Could not find 'points' array in JSON file" << std::endl;
        return edge_points;
    }
    
    // Find the opening bracket of the points array
    size_t array_start = json_content.find('[', points_start);
    if (array_start == std::string::npos) {
        std::cerr << "Error: Could not find points array start" << std::endl;
        return edge_points;
    }
    
    // Extract the points array content
    int bracket_count = 0;
    size_t array_end = array_start;
    bool in_array = false;
    
    for (size_t i = array_start; i < json_content.length(); i++) {
        if (json_content[i] == '[') {
            bracket_count++;
            in_array = true;
        } else if (json_content[i] == ']') {
            bracket_count--;
            if (bracket_count == 0 && in_array) {
                array_end = i;
                break;
            }
        }
    }
    
    std::string points_array = json_content.substr(array_start, array_end - array_start + 1);
    
    // Parse individual point pairs [x, y]
    // Use regex to find all [number, number] patterns
    std::regex point_regex(R"(\[\s*([0-9]+\.[0-9]+|[0-9]+)\s*,\s*([0-9]+\.[0-9]+|[0-9]+)\s*\])");
    std::sregex_iterator iter(points_array.begin(), points_array.end(), point_regex);
    std::sregex_iterator end;
    
    for (; iter != end; ++iter) {
        std::smatch match = *iter;
        if (match.size() == 3) {
            try {
                double x = std::stod(match[1].str());
                double y = std::stod(match[2].str());
                edge_points.push_back(EdgePosition(x, y));
            } catch (const std::exception& e) {
                std::cerr << "Warning: Could not parse point: " << match[0] << std::endl;
            }
        }
    }
    
    std::cout << "Loaded " << edge_points.size() << " edge points from JSON file" << std::endl;
    return edge_points;
}

cv::Mat display_edge_points_on_image(
    const cv::Mat& image,
    const std::vector<EdgePosition>& edge_points,
    const cv::Scalar& point_color,
    int point_radius,
    bool show_image,
    const std::string& window_name
) {
    // Convert image to BGR if it's grayscale (for color visualization)
    cv::Mat display_image;
    if (image.channels() == 1) {
        cv::cvtColor(image, display_image, cv::COLOR_GRAY2BGR);
    } else {
        display_image = image.clone();
    }
    
    // Convert to 8-bit if needed
    if (display_image.depth() != CV_8U) {
        cv::Mat temp;
        display_image.convertTo(temp, CV_8U);
        display_image = temp;
    }
    
    // Draw each edge point
    for (const auto& edge : edge_points) {
        cv::Point center(static_cast<int>(std::round(edge.x)), static_cast<int>(std::round(edge.y)));
        
        // Check bounds
        if (center.x >= 0 && center.x < display_image.cols && 
            center.y >= 0 && center.y < display_image.rows) {
            cv::circle(display_image, center, point_radius, point_color, -1);  // Filled circle
        }
    }
    
    if (show_image) {
        cv::imshow(window_name, display_image);
        std::cout << "Displaying image with " << edge_points.size() << " edge points. Press any key to continue..." << std::endl;
        cv::waitKey(0);
    }
    
    return display_image;
}

cv::Mat display_edge_detection_result(
    const cv::Mat& image,
    const std::vector<EdgePosition>& edge_points,
    const std::vector<PixelCoordinate>& origin_points,
    const cv::Scalar& point_color,
    const cv::Scalar& vector_color,
    int point_radius,
    bool show_image,
    const std::string& window_name
) {
    // Convert image to BGR if it's grayscale
    cv::Mat display_image;
    if (image.channels() == 1) {
        cv::cvtColor(image, display_image, cv::COLOR_GRAY2BGR);
    } else {
        display_image = image.clone();
    }
    
    // Convert to 8-bit if needed
    if (display_image.depth() != CV_8U) {
        cv::Mat temp;
        display_image.convertTo(temp, CV_8U);
        display_image = temp;
    }
    
    // Draw vectors if origin points are provided
    bool draw_vectors = !origin_points.empty() && origin_points.size() == edge_points.size();
    
    if (draw_vectors) {
        for (size_t i = 0; i < edge_points.size(); i++) {
            cv::Point origin = origin_points[i].toPoint();
            cv::Point2d edge_pt = edge_points[i].toPoint2d();
            cv::Point edge_int(static_cast<int>(std::round(edge_pt.x)), 
                              static_cast<int>(std::round(edge_pt.y)));
            
            // Draw vector (arrow) from origin to edge
            if (origin.x >= 0 && origin.x < display_image.cols && 
                origin.y >= 0 && origin.y < display_image.rows &&
                edge_int.x >= 0 && edge_int.x < display_image.cols && 
                edge_int.y >= 0 && edge_int.y < display_image.rows) {
                cv::arrowedLine(display_image, origin, edge_int, vector_color, 1, cv::LINE_AA, 0, 0.3);
            }
        }
    }
    
    // Draw edge points
    for (const auto& edge : edge_points) {
        cv::Point center(static_cast<int>(std::round(edge.x)), 
                        static_cast<int>(std::round(edge.y)));
        
        if (center.x >= 0 && center.x < display_image.cols && 
            center.y >= 0 && center.y < display_image.rows) {
            cv::circle(display_image, center, point_radius, point_color, -1);
        }
    }
    
    if (show_image) {
        cv::imshow(window_name, display_image);
        std::cout << "Displaying image with " << edge_points.size() << " edge points";
        if (draw_vectors) {
            std::cout << " and vectors";
        }
        std::cout << ". Press any key to continue..." << std::endl;
        cv::waitKey(0);
    }
    
    return display_image;
}

