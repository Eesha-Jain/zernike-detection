#include "image_utils.h"
#include <iostream>
#include <limits>

const double SMALL = std::numeric_limits<double>::epsilon();

// Helper function to prevent division by zero
cv::Mat zero_to_small(const cv::Mat& A) {
    cv::Mat result = A.clone();
    double small = SMALL;
    
    for (int i = 0; i < result.rows; i++) {
        for (int j = 0; j < result.cols; j++) {
            double val = result.at<double>(i, j);
            if (val < small && val >= 0) {
                result.at<double>(i, j) = small;
            } else if (val > -small && val < 0) {
                result.at<double>(i, j) = -small;
            }
        }
    }
    return result;
}

// Read image as grayscale
cv::Mat read_image(const std::string& filename) {
    cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: Could not read image " << filename << std::endl;
        return cv::Mat();
    }
    // Convert to double precision for calculations
    cv::Mat img_double;
    img.convertTo(img_double, CV_64F);
    return img_double;
}

// Apply Gaussian blur to image
cv::Mat blur_image(const cv::Mat& img, int Ks, double strength) {
    int kernel_size = Ks;
    if (kernel_size % 2 != 1) {
        std::cout << "blur_image: Ks must be odd! Continuing with Ks = Ks-1" << std::endl;
        kernel_size = Ks - 1;
    }
    
    cv::Mat blurred;
    cv::GaussianBlur(img, blurred, cv::Size(kernel_size, kernel_size), strength);
    return blurred;
}

