# Build and Run Script for Zernike Edge Detection
# Usage: .\build_and_run.ps1 [image_file] [json_file]

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Zernike Edge Detection - Build & Run" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Create build directory if it doesn't exist
if (-not (Test-Path build)) {
    Write-Host "Creating build directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path build | Out-Null
}

Set-Location build

# Clean previous build if CMakeCache exists (optional - comment out if you want incremental builds)
if (Test-Path CMakeCache.txt) {
    Write-Host "Cleaning previous build..." -ForegroundColor Yellow
    Remove-Item CMakeCache.txt, CMakeFiles -Recurse -Force -ErrorAction SilentlyContinue
}

# Configure CMake
Write-Host "Configuring CMake..." -ForegroundColor Yellow
$cmakeSuccess = $false

# Try auto-detect first (most flexible)
Write-Host "  Trying auto-detect..." -ForegroundColor Yellow
$result = & cmake .. 2>&1
if ($LASTEXITCODE -eq 0) {
    $cmakeSuccess = $true
    Write-Host "  Auto-detected generator" -ForegroundColor Green
} else {
    # Try MinGW (common on Windows without Visual Studio)
    Write-Host "  Trying MinGW..." -ForegroundColor Yellow
    $result = & cmake .. -G "MinGW Makefiles" 2>&1
    if ($LASTEXITCODE -eq 0) {
        $cmakeSuccess = $true
        Write-Host "  Using MinGW Makefiles" -ForegroundColor Green
    } else {
        # Try Visual Studio 2022
        Write-Host "  Trying Visual Studio 2022..." -ForegroundColor Yellow
        $result = & cmake .. -G "Visual Studio 17 2022" 2>&1
        if ($LASTEXITCODE -eq 0) {
            $cmakeSuccess = $true
            Write-Host "  Using Visual Studio 17 2022" -ForegroundColor Green
        } else {
            # Try Visual Studio 2019
            Write-Host "  Trying Visual Studio 2019..." -ForegroundColor Yellow
            $result = & cmake .. -G "Visual Studio 16 2019" 2>&1
            if ($LASTEXITCODE -eq 0) {
                $cmakeSuccess = $true
                Write-Host "  Using Visual Studio 16 2019" -ForegroundColor Green
            }
        }
    }
}

if (-not $cmakeSuccess) {
    Write-Host ""
    Write-Host "CMake configuration failed!" -ForegroundColor Red
    Write-Host $result -ForegroundColor Red
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host "NO C++ COMPILER FOUND" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "You need to install a C++ compiler:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Option 1: MinGW-w64 (Recommended)" -ForegroundColor Cyan
    Write-Host "  1. Download MSYS2: https://www.msys2.org/" -ForegroundColor White
    Write-Host "  2. Install, then run in MSYS2 terminal:" -ForegroundColor White
    Write-Host "     pacman -Syu" -ForegroundColor Gray
    Write-Host "     pacman -S mingw-w64-x86_64-gcc" -ForegroundColor Gray
    Write-Host "     pacman -S mingw-w64-x86_64-cmake" -ForegroundColor Gray
    Write-Host "  3. Add C:\msys64\mingw64\bin to PATH" -ForegroundColor White
    Write-Host ""
    Write-Host "Option 2: Visual Studio Build Tools" -ForegroundColor Cyan
    Write-Host "  1. Download: https://visualstudio.microsoft.com/downloads/" -ForegroundColor White
    Write-Host "  2. Install 'Desktop development with C++'" -ForegroundColor White
    Write-Host ""
    Write-Host "After installing, restart PowerShell and try again!" -ForegroundColor Green
    Write-Host ""
    Write-Host "See INSTALL_COMPILER.md for detailed instructions." -ForegroundColor Gray
    exit 1
}

# Build
Write-Host ""
Write-Host "Building project..." -ForegroundColor Yellow
$buildResult = & cmake --build . --config Release 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    Write-Host $buildResult -ForegroundColor Red
    exit 1
}

Write-Host "Build successful!" -ForegroundColor Green
Write-Host ""

# Determine executable path
$exePath = "Release\zernike_edge_detection.exe"
if (-not (Test-Path $exePath)) {
    $exePath = "zernike_edge_detection.exe"
    if (-not (Test-Path $exePath)) {
        Write-Host "Executable not found!" -ForegroundColor Red
        Write-Host "Expected: $exePath" -ForegroundColor Red
        exit 1
    }
}

# Get file arguments
$imageFile = "..\images\moon.jpg"
$jsonFile = "..\chosen_points.json"

if ($args.Length -ge 2) {
    $imageFile = $args[0]
    $jsonFile = $args[1]
} elseif ($args.Length -eq 1) {
    Write-Host "Warning: Need both image and JSON file. Using defaults." -ForegroundColor Yellow
}

# Check if files exist
if (-not (Test-Path $imageFile)) {
    Write-Host "Error: Image file not found: $imageFile" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $jsonFile)) {
    Write-Host "Error: JSON file not found: $jsonFile" -ForegroundColor Red
    exit 1
}

# Run
Write-Host "Running program..." -ForegroundColor Green
Write-Host "  Image: $imageFile" -ForegroundColor Cyan
Write-Host "  JSON:  $jsonFile" -ForegroundColor Cyan
Write-Host ""

& $exePath $imageFile $jsonFile

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Program exited with error code: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Program completed successfully!" -ForegroundColor Green

