#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <opencv2/opencv.hpp>
#include <string>

// Constants
extern const double SMALL;

// Image I/O functions
cv::Mat read_image(const std::string& filename);
cv::Mat blur_image(const cv::Mat& img, int Ks, double strength = 1.0);

// Numerical utilities
cv::Mat zero_to_small(const cv::Mat& A);

#endif // IMAGE_UTILS_H

