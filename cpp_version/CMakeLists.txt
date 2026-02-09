cmake_minimum_required(VERSION 3.10)
project(ZernikeEdgeDetection)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

# Create executable
add_executable(zernike_edge_detection ${SOURCES})

# Link OpenCV libraries
target_link_libraries(zernike_edge_detection ${OpenCV_LIBS})

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

