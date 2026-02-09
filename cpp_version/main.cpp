#include "zernike_edge_detection.h"
#include "image_utils.h"
#include "file_utils.h"
#include "helper.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>

// Filesystem compatibility
#if __cplusplus >= 201703L && defined(__has_include)
    #if __has_include(<filesystem>)
        #include <filesystem>
        namespace fs = std::filesystem;
    #elif __has_include(<experimental/filesystem>)
        #include <experimental/filesystem>
        namespace fs = std::experimental::filesystem;
    #else
        // Fallback to manual path handling
        #define NO_FILESYSTEM
    #endif
#else
    #define NO_FILESYSTEM
#endif

std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d_%H-%M-%S");
    return ss.str();
}

int main(int argc, char* argv[]) {
    std::cout << "Zernike Edge Detection - C++ Version" << std::endl;
    std::cout << "====================================" << std::endl;
    
    // Read settings files
    std::string settings_dir = "Settings";
    
    // Read image info
    #ifdef NO_FILESYSTEM
    std::string imageinfo_path = path_join(settings_dir, "imageinfo.csv");
    #else
    std::string imageinfo_path = (fs::path(settings_dir) / "imageinfo.csv").string();
    #endif
    std::vector<ImageInfo> img_info = read_csv(imageinfo_path);
    
    if (img_info.empty()) {
        std::cerr << "Error: No images to process or could not read imageinfo.csv" << std::endl;
        return 1;
    }
    
    int ncases = img_info.size();
    std::cout << "Found " << ncases << " images to process" << std::endl;
    
    // Read save directory
    #ifdef NO_FILESYSTEM
    std::string save_path = path_join(settings_dir, "saveTo.csv");
    #else
    std::string save_path = (fs::path(settings_dir) / "saveTo.csv").string();
    #endif
    std::string save_dir = read_single_value_csv(save_path, "directory");
    
    if (save_dir.empty()) {
        std::cerr << "Error: Could not read save directory" << std::endl;
        return 1;
    }
    
    // Create save directory if it doesn't exist
    #ifdef NO_FILESYSTEM
    if (!path_exists(save_dir)) {
        create_directories(save_dir);
    }
    #else
    if (!fs::exists(save_dir)) {
        fs::create_directories(save_dir);
    }
    #endif
    
    // Create timestamped subdirectory
    std::string timestamp = get_timestamp();
    #ifdef NO_FILESYSTEM
    std::string savedirect = path_join(save_dir, timestamp);
    create_directories(savedirect);
    #else
    fs::path savedirect = fs::path(save_dir) / timestamp;
    fs::create_directories(savedirect);
    #endif
    
    std::cout << "Results will be saved to: " << 
        #ifdef NO_FILESYSTEM
        savedirect
        #else
        savedirect.string()
        #endif
        << std::endl;
    
    // Read data folder
    #ifdef NO_FILESYSTEM
    std::string datafolder_path = path_join(settings_dir, "dataFolder.csv");
    #else
    std::string datafolder_path = (fs::path(settings_dir) / "dataFolder.csv").string();
    #endif
    std::string datafolder = read_single_value_csv(datafolder_path, "directory");
    
    if (datafolder.empty()) {
        std::cerr << "Error: Could not read data folder" << std::endl;
        return 1;
    }
    
    // Process each image
    for (int c = 0; c < ncases; c++) {
        std::cout << "\nProcessing image " << (c + 1) << " of " << ncases << ": " << img_info[c].name << std::endl;
        
        // Construct full image path
        #ifdef NO_FILESYSTEM
        std::string image_path = path_join(datafolder, img_info[c].name);
        #else
        std::string image_path = (fs::path(datafolder) / img_info[c].name).string();
        #endif
        
        // Read parameters
        #ifdef NO_FILESYSTEM
        std::string pars_path = path_join(settings_dir, img_info[c].gpars);
        #else
        std::string pars_path = (fs::path(settings_dir) / img_info[c].gpars).string();
        #endif
        std::vector<double> pars = read_external_data(pars_path);
        
        if (pars.size() < 7) {
            std::cerr << "Error: Invalid parameter file for " << img_info[c].name << std::endl;
            continue;
        }
        
        int K_s = static_cast<int>(pars[0]);
        double k_min = pars[1];
        double k_max = pars[2];
        double l_max = pars[3];
        double phi_min = pars[4];
        double outlier_sigma = pars[5];
        double blur = pars[6];
        
        // Load image
        cv::Mat img_o = read_image(image_path);
        if (img_o.empty()) {
            std::cerr << "Error: Could not load image " << image_path << std::endl;
            continue;
        }
        
        // Blur image
        cv::Mat img_f = blur_image(img_o, static_cast<int>(blur));
        
        // Detect edges
        std::cout << "  Detecting edges with K_s=" << K_s << ", k_min=" << k_min 
                  << ", k_max=" << k_max << ", l_max=" << l_max << ", phi_min=" << phi_min << std::endl;
        
        EdgeResult edge_result = ghosal_edge_v2(
            img_f, K_s, k_min, k_max, l_max, phi_min, true, false, true
        );
        
        std::cout << "  Found " << edge_result.edges.size() << " edge points" << std::endl;
        
        // Save results
        #ifdef NO_FILESYSTEM
        std::string basename = path_stem(img_info[c].name);
        std::string savename = path_join(savedirect, basename + ".txt");
        #else
        std::string basename = fs::path(img_info[c].name).stem().string();
        std::string savename = (savedirect / (basename + ".txt")).string();
        #endif
        save_edge_results(edge_result, savename);
        
        std::cout << "  Saved to: " << savename << std::endl;
    }
    
    std::cout << "\nProcessing complete!" << std::endl;
    std::cout << "Press ENTER to exit...";
    std::cin.get();
    
    return 0;
}

