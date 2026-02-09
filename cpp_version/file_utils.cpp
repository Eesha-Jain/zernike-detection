#include "file_utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <iomanip>

// Read external data file (tab or space separated)
std::vector<double> read_external_data(const std::string& filename, char sep) {
    std::vector<double> data;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        while (std::getline(iss, token, sep)) {
            if (!token.empty()) {
                try {
                    data.push_back(std::stod(token));
                } catch (const std::exception& e) {
                    // Skip invalid tokens
                }
            }
        }
    }
    
    file.close();
    return data;
}

// Simple CSV reader helper
std::vector<ImageInfo> read_csv(const std::string& filename) {
    std::vector<ImageInfo> results;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open CSV file " << filename << std::endl;
        return results;
    }
    
    std::string line;
    bool first_line = true;
    
    while (std::getline(file, line)) {
        if (first_line) {
            first_line = false;
            continue;  // Skip header
        }
        
        std::istringstream iss(line);
        std::string token;
        ImageInfo info;
        int col = 0;
        
        while (std::getline(iss, token, ',')) {
            // Remove quotes and whitespace
            token.erase(0, token.find_first_not_of(" \t\""));
            token.erase(token.find_last_not_of(" \t\"") + 1);
            
            if (col == 0) info.name = token;
            else if (col == 1) info.gpars = token;
            else if (col == 2) info.process = token;
            col++;
        }
        
        if (info.process == "x") {
            results.push_back(info);
        }
    }
    
    file.close();
    return results;
}

std::string read_single_value_csv(const std::string& filename, const std::string& column_name) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open CSV file " << filename << std::endl;
        return "";
    }
    
    std::string line;
    std::getline(file, line);  // Read header
    std::getline(file, line);   // Read data line
    
    // Try tab first, then comma
    std::istringstream iss(line);
    std::string token;
    if (line.find('\t') != std::string::npos) {
        std::getline(iss, token, '\t');  // Tab-separated
    } else {
        std::getline(iss, token, ',');   // Comma-separated
    }
    
    // Remove quotes and whitespace
    token.erase(0, token.find_first_not_of(" \t\""));
    token.erase(token.find_last_not_of(" \t\"") + 1);
    
    file.close();
    return token;
}

#ifdef NO_FILESYSTEM
// Simple path join helper
std::string path_join(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    if (a.back() == '/' || a.back() == '\\') return a + b;
    #ifdef _WIN32
        return a + "\\" + b;
    #else
        return a + "/" + b;
    #endif
}

std::string path_stem(const std::string& path) {
    size_t last_slash = path.find_last_of("/\\");
    size_t last_dot = path.find_last_of(".");
    if (last_dot != std::string::npos && last_dot > last_slash) {
        return path.substr(last_slash + 1, last_dot - last_slash - 1);
    }
    return path.substr(last_slash + 1);
}

bool path_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

void create_directories(const std::string& path) {
    // Simple implementation - create parent directories if needed
    #ifdef _WIN32
        system(("mkdir \"" + path + "\" 2>nul").c_str());
    #else
        system(("mkdir -p \"" + path + "\" 2>/dev/null").c_str());
    #endif
}
#endif

std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d_%H-%M-%S");
    return ss.str();
}

