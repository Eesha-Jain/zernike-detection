# Simple Compile and Run Script
# This script compiles the code directly with g++ (no CMake needed)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Zernike Edge Detection - Simple Build" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Check if g++ is available
$gpp = Get-Command g++ -ErrorAction SilentlyContinue
if (-not $gpp) {
    Write-Host "ERROR: g++ compiler not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install MinGW-w64:" -ForegroundColor Yellow
    Write-Host "  1. Download MSYS2: https://www.msys2.org/" -ForegroundColor White
    Write-Host "  2. Install, then run in MSYS2 terminal:" -ForegroundColor White
    Write-Host "     pacman -Syu" -ForegroundColor Gray
    Write-Host "     pacman -S mingw-w64-x86_64-gcc" -ForegroundColor Gray
    Write-Host "  3. Add C:\msys64\mingw64\bin to PATH" -ForegroundColor White
    Write-Host ""
    exit 1
}

Write-Host "Found g++ compiler" -ForegroundColor Green

# Try to find OpenCV
$opencvFound = $false
$opencvInclude = ""
$opencvLib = ""
$opencvBin = ""

# Check environment variable first
if ($env:OPENCV_DIR) {
    $envPath = $env:OPENCV_DIR
    if (Test-Path "$envPath\include\opencv2") {
        $opencvInclude = "$envPath\include"
        if (Test-Path "$envPath\x64\mingw\lib") {
            $opencvLib = "$envPath\x64\mingw\lib"
            $opencvBin = "$envPath\x64\mingw\bin"
        } elseif (Test-Path "$envPath\lib") {
            $opencvLib = "$envPath\lib"
            $opencvBin = "$envPath\bin"
        }
        if ($opencvLib -ne "" -and (Test-Path $opencvLib)) {
            $opencvFound = $true
            Write-Host "Found OpenCV via OPENCV_DIR: $envPath" -ForegroundColor Green
        }
    }
}

# Common OpenCV locations on Windows
if (-not $opencvFound) {
    $opencvPaths = @(
        "C:\opencv\build",
        "C:\opencv",
        "$env:USERPROFILE\opencv\build",
        "$env:USERPROFILE\opencv",
        "C:\vcpkg\installed\x64-windows",
        "C:\tools\opencv\build",
        "C:\Program Files\opencv\build"
    )

    foreach ($path in $opencvPaths) {
        if (Test-Path "$path\include\opencv2") {
            $opencvInclude = "$path\include"
            if (Test-Path "$path\x64\mingw\lib") {
                $opencvLib = "$path\x64\mingw\lib"
                $opencvBin = "$path\x64\mingw\bin"
            } elseif (Test-Path "$path\lib") {
                $opencvLib = "$path\lib"
                $opencvBin = "$path\bin"
            } elseif (Test-Path "$path\x64\vc16\lib") {
                $opencvLib = "$path\x64\vc16\lib"
                $opencvBin = "$path\x64\vc16\bin"
            }
            
            if ($opencvLib -ne "" -and (Test-Path $opencvLib)) {
                $opencvFound = $true
                Write-Host "Found OpenCV at: $path" -ForegroundColor Green
                break
            }
        }
    }
}

# Also try pkg-config (if available)
if (-not $opencvFound) {
    $pkgConfig = Get-Command pkg-config -ErrorAction SilentlyContinue
    if ($pkgConfig) {
        try {
            $cflags = pkg-config --cflags opencv4 2>$null
            $libs = pkg-config --libs-only-L opencv4 2>$null
            
            if ($cflags) {
                # Extract include path (look for -I flag)
                $includeMatch = [regex]::Match($cflags, '-I([^\s]+)')
                if ($includeMatch.Success) {
                    $opencvInclude = $includeMatch.Groups[1].Value
                    # Convert forward slashes to backslashes
                    $opencvInclude = $opencvInclude -replace '/', '\'
                    # Resolve relative paths (handles .. and .)
                    if ($opencvInclude -match '\.\.|^\.') {
                        $resolved = try { (Resolve-Path $opencvInclude -ErrorAction Stop).Path } catch { $opencvInclude }
                        $opencvInclude = $resolved
                    }
                }
                
                # Extract library path (look for -L flag)
                if ($libs) {
                    $libMatch = [regex]::Match($libs, '-L([^\s]+)')
                    if ($libMatch.Success) {
                        $opencvLib = $libMatch.Groups[1].Value
                        # Convert forward slashes to backslashes
                        $opencvLib = $opencvLib -replace '/', '\'
                        # Resolve relative paths (handles .. and .)
                        if ($opencvLib -match '\.\.|^\.') {
                            $resolved = try { (Resolve-Path $opencvLib -ErrorAction Stop).Path } catch { $opencvLib }
                            $opencvLib = $resolved
                        }
                        # Also try to find bin directory
                        $opencvBin = $opencvLib -replace '\\lib$', '\bin'
                        if (-not (Test-Path $opencvBin)) {
                            $opencvBin = $opencvLib -replace '\\lib$', ''
                        }
                    }
                }
                
                if ($opencvInclude -and (Test-Path $opencvInclude)) {
                    $opencvFound = $true
                    Write-Host "Found OpenCV via pkg-config" -ForegroundColor Green
                }
            }
        } catch {
            # pkg-config failed, continue
        }
    }
}

if (-not $opencvFound) {
    Write-Host ""
    Write-Host "ERROR: OpenCV not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install OpenCV:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Option 1: Using vcpkg (Recommended)" -ForegroundColor Cyan
    Write-Host "  1. Install vcpkg: https://github.com/microsoft/vcpkg" -ForegroundColor White
    Write-Host "  2. Run: vcpkg install opencv4:x64-windows" -ForegroundColor White
    Write-Host ""
    Write-Host "Option 2: Download pre-built binaries" -ForegroundColor Cyan
    Write-Host "  1. Download from: https://opencv.org/releases/" -ForegroundColor White
    Write-Host "  2. Extract to C:\opencv" -ForegroundColor White
    Write-Host ""
    Write-Host "Option 3: Use CMake method (see README.md)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "After installing, you can:" -ForegroundColor Yellow
    Write-Host "  - Set OPENCV_DIR environment variable to OpenCV path" -ForegroundColor White
    Write-Host "  - Or place OpenCV in one of these locations:" -ForegroundColor White
    foreach ($path in $opencvPaths) {
        Write-Host "    - $path" -ForegroundColor Gray
    }
    Write-Host ""
    exit 1
}

# Compile
Write-Host ""
Write-Host "Compiling..." -ForegroundColor Yellow

$sourceFiles = @(
    "zernike_edge_detection.cpp",
    "image_utils.cpp",
    "file_utils.cpp",
    "helper.cpp",
    "main.cpp"
)

# Check all source files exist
foreach ($file in $sourceFiles) {
    if (-not (Test-Path $file)) {
        Write-Host "ERROR: Source file not found: $file" -ForegroundColor Red
        exit 1
    }
}

# Build compile command
$compileCmd = "g++ -std=c++17 -O2"
$compileCmd += " -I`"$opencvInclude`""

# Find OpenCV libraries
$opencvLibs = @()
$libFiles = Get-ChildItem -Path $opencvLib -Filter "*.a" -ErrorAction SilentlyContinue
if ($libFiles) {
    # MinGW libraries (.a or .dll.a files)
    # Remove 'lib' prefix and '.dll.a' or '.a' suffix to get library name
    $libNames = $libFiles | ForEach-Object { 
        $name = $_.BaseName -replace '^lib', ''
        # Remove .dll if present (for .dll.a files)
        $name = $name -replace '\.dll$', ''
        $name
    }
    $requiredLibs = @('core', 'imgproc', 'imgcodecs', 'highgui')
    foreach ($reqLib in $requiredLibs) {
        $found = $libNames | Where-Object { $_ -match "^opencv_$reqLib`$" } | Select-Object -First 1
        if ($found) {
            $opencvLibs += "-l$found"
        }
    }
    # If we didn't find all, use default names (linker will find them)
    if ($opencvLibs.Count -lt 4) {
        $opencvLibs = @("-lopencv_core", "-lopencv_imgproc", "-lopencv_imgcodecs", "-lopencv_highgui")
    }
} else {
    # Try .lib files (Visual Studio) or default names
    $opencvLibs = @("-lopencv_core", "-lopencv_imgproc", "-lopencv_imgcodecs", "-lopencv_highgui")
}

$compileCmd += " -L`"$opencvLib`""
$compileCmd += " " + ($opencvLibs -join " ")
$compileCmd += " " + ($sourceFiles -join " ")
$compileCmd += " -o zernike_edge_detection.exe"

Write-Host "Running: $compileCmd" -ForegroundColor Gray
Write-Host ""

$result = Invoke-Expression $compileCmd 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Compilation failed!" -ForegroundColor Red
    Write-Host $result -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  - Make sure OpenCV is properly installed" -ForegroundColor White
    Write-Host "  - Check that all source files are present" -ForegroundColor White
    Write-Host "  - Try using CMake method instead (see README.md)" -ForegroundColor White
    exit 1
}

Write-Host "Compilation successful!" -ForegroundColor Green
Write-Host ""

# Check if executable was created
if (-not (Test-Path "zernike_edge_detection.exe")) {
    Write-Host "ERROR: Executable not created!" -ForegroundColor Red
    exit 1
}

# Get file arguments
$imageFile = "images\moon.jpg"
$jsonFile = "chosen_points.json"

if ($args.Length -ge 2) {
    $imageFile = $args[0]
    $jsonFile = $args[1]
} elseif ($args.Length -eq 1) {
    Write-Host "Warning: Need both image and JSON file. Using defaults." -ForegroundColor Yellow
}

# Check if files exist
if (-not (Test-Path $imageFile)) {
    Write-Host "Error: Image file not found: $imageFile" -ForegroundColor Red
    Write-Host "Usage: .\compile_and_run.ps1 [image_file] [json_file]" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path $jsonFile)) {
    Write-Host "Error: JSON file not found: $jsonFile" -ForegroundColor Red
    Write-Host "Usage: .\compile_and_run.ps1 [image_file] [json_file]" -ForegroundColor Yellow
    exit 1
}

# Copy OpenCV DLLs if needed (for MinGW)
if ($opencvBin -ne "" -and (Test-Path $opencvBin)) {
    $dlls = Get-ChildItem -Path $opencvBin -Filter "*.dll" -ErrorAction SilentlyContinue
    foreach ($dll in $dlls) {
        if ($dll.Name -match 'opencv') {
            Copy-Item $dll.FullName -Destination "." -Force -ErrorAction SilentlyContinue
        }
    }
}

# Run
Write-Host "Running program..." -ForegroundColor Green
Write-Host "  Image: $imageFile" -ForegroundColor Cyan
Write-Host "  JSON:  $jsonFile" -ForegroundColor Cyan
Write-Host ""

& .\zernike_edge_detection.exe $imageFile $jsonFile

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Program exited with error code: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Program completed successfully!" -ForegroundColor Green

