#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <vector>
#include <string>

// CSV reading structures
struct ImageInfo {
    std::string name;
    std::string gpars;
    std::string process;
};

// File I/O functions
std::vector<double> read_external_data(const std::string& filename, char sep = '\t');
std::vector<ImageInfo> read_csv(const std::string& filename);
std::string read_single_value_csv(const std::string& filename, const std::string& column_name);

// Utility functions
std::string get_timestamp();

// Path utilities (for filesystem compatibility)
#ifdef NO_FILESYSTEM
std::string path_join(const std::string& a, const std::string& b);
std::string path_stem(const std::string& path);
bool path_exists(const std::string& path);
void create_directories(const std::string& path);
#endif

#endif // FILE_UTILS_H

