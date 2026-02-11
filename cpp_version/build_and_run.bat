@echo off
REM Build and Run Script for Zernike Edge Detection
REM Usage: build_and_run.bat [image_file] [json_file]

echo ========================================
echo Zernike Edge Detection - Build ^& Run
echo ========================================
echo.

cd /d "%~dp0"

REM Create build directory if it doesn't exist
if not exist build mkdir build
cd build

REM Clean previous build (optional - remove this section for incremental builds)
if exist CMakeCache.txt (
    echo Cleaning previous build...
    del /Q CMakeCache.txt 2>nul
    rmdir /S /Q CMakeFiles 2>nul
)

REM Configure CMake
echo Configuring CMake...
cmake .. -G "Visual Studio 17 2022" >nul 2>&1
if errorlevel 1 (
    cmake .. -G "Visual Studio 16 2019" >nul 2>&1
    if errorlevel 1 (
        cmake .. -G "MinGW Makefiles" >nul 2>&1
        if errorlevel 1 (
            echo Trying auto-detect...
            cmake ..
        )
    )
)

if errorlevel 1 (
    echo.
    echo CMake configuration failed!
    echo.
    echo Please install:
    echo   1. CMake (https://cmake.org/download/)
    echo   2. Visual Studio Build Tools or MinGW
    pause
    exit /b 1
)

REM Build
echo.
echo Building project...
cmake --build . --config Release
if errorlevel 1 (
    echo.
    echo Build failed!
    pause
    exit /b 1
)

echo Build successful!
echo.

REM Determine executable path
set EXE_PATH=Release\zernike_edge_detection.exe
if not exist "%EXE_PATH%" (
    set EXE_PATH=zernike_edge_detection.exe
    if not exist "%EXE_PATH%" (
        echo Executable not found!
        pause
        exit /b 1
    )
)

REM Get file arguments
set IMAGE_FILE=..\images\moon.jpg
set JSON_FILE=..\chosen_points.json

if not "%1"=="" set IMAGE_FILE=%1
if not "%2"=="" set JSON_FILE=%2

REM Check if files exist
if not exist "%IMAGE_FILE%" (
    echo Error: Image file not found: %IMAGE_FILE%
    pause
    exit /b 1
)

if not exist "%JSON_FILE%" (
    echo Error: JSON file not found: %JSON_FILE%
    pause
    exit /b 1
)

REM Run
echo Running program...
echo   Image: %IMAGE_FILE%
echo   JSON:  %JSON_FILE%
echo.

"%EXE_PATH%" "%IMAGE_FILE%" "%JSON_FILE%"

if errorlevel 1 (
    echo.
    echo Program exited with error code: %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Program completed successfully!
pause


