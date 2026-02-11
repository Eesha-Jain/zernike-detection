# Zernike Edge Detection - C++ Implementation

This is a C++ implementation of the Zernike moments sub-pixel edge detection algorithm based on Christian (2017) "Accurate Planetary Limb Localization."

## Quick Start - Easiest Way to Run

### Prerequisites

1. **C++ Compiler (g++)**: Install MinGW-w64
   - Download MSYS2 from https://www.msys2.org/
   - Install it, then open MSYS2 terminal and run:
     ```bash
     pacman -Syu
     pacman -S mingw-w64-x86_64-gcc
     ```
   - Add `C:\msys64\mingw64\bin` to your Windows PATH

2. **OpenCV**: Install OpenCV library
    ```powershell
    # If using MSYS2
    pacman -S mingw-w64-x86_64-opencv
    ```

### Step 1: Open PowerShell in the project folder

Navigate to the `cpp_version` folder:
```powershell
cd cpp_version
```

### Step 2: Run the compile script

Simply run:
```powershell
.\compile_and_run.ps1
```

This will:
- Automatically find your compiler and OpenCV
- Compile all the source files
- Run the program with default files (`images\moon.jpg` and `chosen_points.json`)

### Step 3: Run with custom files (optional)

To use different image and JSON files:
```powershell
.\compile_and_run.ps1 "path\to\image.jpg" "path\to\points.json"
```

### That's it!

The program will:
1. Load your image
2. Load edge points from the JSON file
3. Refine the edge positions using Zernike moments
4. Display the results (green = initial points, blue = refined points)

Press any key in the image window to close it.

---

## Alternative: Using CMake (if simple script doesn't work)

If the simple compile script doesn't work for you, you can use CMake:

### Step 1: Create build directory
```powershell
mkdir build
cd build
```

### Step 2: Configure with CMake
```powershell
cmake .. -G "MinGW Makefiles"
```
(Or use `"Visual Studio 17 2022"` if you have Visual Studio)

### Step 3: Build
```powershell
cmake --build . --config Release
```

### Step 4: Run
```powershell
.\Release\zernike_edge_detection.exe ..\images\moon.jpg ..\chosen_points.json
```

---

## Input File Format

The JSON file should contain edge points in this format:
```json
{
  "points": [
    [x1, y1],
    [x2, y2],
    ...
  ],
  "count": N
}
```

---

## Troubleshooting

### "g++ not found"
- Make sure MinGW-w64 is installed
- Add `C:\msys64\mingw64\bin` to your PATH
- Restart PowerShell after adding to PATH

### "OpenCV not found"
- Install OpenCV using one of the methods above
- The script looks in common locations:
  - `C:\opencv\build`
  - `C:\opencv`
  - `C:\vcpkg\installed\x64-windows`
- You can also set the `OPENCV_DIR` environment variable

### "Compilation failed"
- Make sure all source files are present
- Check that OpenCV is properly installed
- Try the CMake method instead

### "DLL not found" at runtime
- Copy OpenCV DLLs to the same folder as the executable
- Or add OpenCV bin directory to your PATH

---

## Project Structure

```
cpp_version/
├── main.cpp                    # Main program
├── zernike_edge_detection.h/cpp # Core algorithm
├── image_utils.h/cpp           # Image I/O utilities
├── file_utils.h/cpp            # File utilities
├── helper.h/cpp                # JSON parsing & visualization
├── compile_and_run.ps1         # Simple build script (USE THIS!)
├── CMakeLists.txt              # CMake configuration (alternative)
└── README.md                   # This file
```

---

## Algorithm Parameters

The algorithm uses these default parameters (in `main.cpp`):
- **Window size**: 7×7 pixels
- **Transition width**: 1.66 (for Gaussian blur)

These can be modified in `main.cpp` if needed.
