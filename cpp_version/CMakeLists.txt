cmake_minimum_required(VERSION 3.10)
project(ZernikeEdgeDetection VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Output directory for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Source files
set(SOURCES
    zernike_edge_detection.cpp
    image_utils.cpp
    file_utils.cpp
    helper.cpp
    main.cpp
)

# Optional example executables
add_executable(example_zernike_usage 
    example_zernike_usage.cpp
    zernike_edge_detection.cpp
    image_utils.cpp
    file_utils.cpp
    helper.cpp
)
target_link_libraries(example_zernike_usage ${OpenCV_LIBS})

# Create executable
add_executable(zernike_edge_detection ${SOURCES})

# Link OpenCV libraries
target_link_libraries(zernike_edge_detection ${OpenCV_LIBS})

# Set output directory (for VS Code compatibility)
set_target_properties(zernike_edge_detection PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release
)

# Print OpenCV info
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# For Windows, copy DLLs to output directory
if(WIN32)
    add_custom_command(TARGET zernike_edge_detection POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:opencv_world>
        $<TARGET_FILE:opencv_imgproc>
        $<TARGET_FILE:opencv_imgcodecs>
        $<TARGET_FILE:opencv_core>
        $<TARGET_FILE_DIR:zernike_edge_detection>
    )
endif()

