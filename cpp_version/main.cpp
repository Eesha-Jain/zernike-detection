/**
 * @file main.cpp
 * @brief Main program for Zernike moments edge detection
 * 
 * This program:
 * 1. Loads initial edge points from a JSON file
 * 2. Refines edge positions using Zernike moments algorithm
 * 3. Displays the results on the image
 */

#include "zernike_edge_detection.h"
#include "helper.h"
#include "image_utils.h"
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "Zernike Moments Edge Detection" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Parse command line arguments
    std::string image_file;
    std::string json_file;
    int window_size = 7;           // Default window size
    double transition_width = 1.66; // Default transition width (for Gaussian blur)
    
    if (argc >= 3) {
        image_file = argv[1];
        json_file = argv[2];
    } else {
        std::cerr << "Error: Must input values for image_file and json_file" << std::endl;
    }
    
    // STEP 1: Load Image
    std::cout << "\n[Step 1] Loading image: " << image_file << std::endl;
    cv::Mat image = read_image(image_file);
    if (image.empty()) {
        std::cerr << "Error: Could not load image " << image_file << std::endl;
        std::cerr << "Please check that the file exists and is a valid image." << std::endl;
        return 1;
    }
    std::cout << "  ✓ Image loaded successfully" << std::endl;
    std::cout << "  Image size: " << image.cols << " × " << image.rows << " pixels" << std::endl;
    std::cout << "  Image type: CV_64F (double precision)" << std::endl;
    

    // STEP 2: Load Initial Edge Points from JSON
    std::cout << "\n[Step 2] Loading initial edge points from: " << json_file << std::endl;
    std::vector<EdgePosition> initial_points = read_edge_points_from_json(json_file);
    
    if (initial_points.empty()) {
        std::cerr << "Error: No initial edge points loaded from JSON file" << std::endl;
        std::cerr << "Please check that the JSON file exists and has the correct format." << std::endl;
        return 1;
    }
    std::cout << "  ✓ Loaded " << initial_points.size() << " initial edge points" << std::endl;
    

    // STEP 3: Refine Edge Positions with Zernike Moments
    std::cout << "\n[Step 3] Refining edge positions with Zernike moments..." << std::endl;
    std::cout << "  Window size: " << window_size << "×" << window_size << std::endl;
    std::cout << "  Transition width: " << transition_width << std::endl;
    std::cout << "  Processing " << initial_points.size() << " edge points..." << std::endl;
    
    EdgeResult refined_result = refine_edge_positions_with_zernike_moments(
        image,
        initial_points,
        window_size,
        transition_width,
        false  // debug mode
    );
    
    if (refined_result.edges.empty()) {
        std::cerr << "Error: No refined edge points were generated" << std::endl;
        return 1;
    }
    
    std::cout << "  ✓ Refined " << refined_result.edges.size() << " edge points to sub-pixel accuracy" << std::endl;
    
    // Calculate statistics
    if (initial_points.size() == refined_result.edges.size()) {
        double total_displacement = 0.0;
        double max_displacement = 0.0;
        for (size_t i = 0; i < initial_points.size(); i++) {
            double dx = refined_result.edges[i].x - initial_points[i].x;
            double dy = refined_result.edges[i].y - initial_points[i].y;
            double displacement = std::sqrt(dx * dx + dy * dy);
            total_displacement += displacement;
            max_displacement = std::max(max_displacement, displacement);
        }
        double avg_displacement = total_displacement / initial_points.size();
        std::cout << "  Average refinement: " << std::fixed << std::setprecision(3) 
                  << avg_displacement << " pixels" << std::endl;
        std::cout << "  Maximum refinement: " << max_displacement << " pixels" << std::endl;
    }
    
    
    // STEP 4: Display Results on Image
    std::cout << "\n[Step 4] Displaying results on image..." << std::endl;
    
    // Load original image for display (try to load as color, fallback to grayscale)
    cv::Mat display_image = cv::imread(image_file, cv::IMREAD_COLOR);
    if (display_image.empty()) {
        // If color load failed, use the grayscale image and convert to BGR
        display_image = image.clone();
        // Convert from CV_64F to CV_8U first
        cv::Mat image_8u;
        image.convertTo(image_8u, CV_8U);
        cv::cvtColor(image_8u, display_image, cv::COLOR_GRAY2BGR);
    } else {
        // Convert color image to 8-bit if needed
        if (display_image.depth() != CV_8U) {
            cv::Mat temp;
            display_image.convertTo(temp, CV_8U);
            display_image = temp;
        }
    }
    
    // Display initial points in green
    std::cout << "  Drawing initial points (green)..." << std::endl;
    cv::Mat result_image = display_edge_points_on_image(
        display_image,
        initial_points,
        cv::Scalar(0, 255, 0),  // Green for initial points
        3,                        // Point radius
        false,                    // Don't show yet
        "Zernike Edge Detection"  // Window name
    );
    
    // Display refined points in blue (larger, on top)
    std::cout << "  Drawing refined points (blue)..." << std::endl;
    result_image = display_edge_points_on_image(
        result_image,
        refined_result.edges,
        cv::Scalar(255, 0, 0),  // Blue for refined points
        5,                       // Larger point radius
        true,                    // Show image now
        "Zernike Edge Detection - Green: Initial, Blue: Refined"
    );
    
    std::cout << "  ✓ Image displayed" << std::endl;
    std::cout << "  Press any key in the image window to continue..." << std::endl;
    
    return 0;
}
