# C++ Migration Summary: SetiAstroSuite Pro

We have completed a fundamental upgrade to the "engine" of SetiAstroSuite Pro, moving the heaviest computations from Python to C++. Here is what has changed in simple terms.

## The New Engine: `saspro_cpp`

### 1. Star Alignment
- **Before**: Python performed calculations step-by-step.
- **Now**: C++ performs star detection and geometric transformation calculations instantly.
- **Result**: Image alignment is significantly faster and more precise.

### 2. Stacking Suite (Image Integration)
This was the most time-consuming part. During stacking, the software must analyze every single pixel of every photo (millions of calculations) to decide which ones to reject (e.g., satellite trails, cosmic rays).

- **Enhanced Algorithms**:
    - We rewrote the most critical rejection algorithms in C++: **Windsorized Sigma Clipping**, **Kappa-Sigma**, **Trimmed Mean**, and **ESD**.
- **Parallelism**:
    - The new code uses **all processor cores** simultaneously. If you have 16 cores, they all work together to crunch the pixels.
- **"Comet" Integration**:
    - The comet stacking mode also utilizes this accelerated engine.

## What changes?
1.  **Speed**: Waiting times during stacking and alignment will drop drastically.
2.  **Stability**: Offloading the workload to C++ reduces unstable memory usage that Python sometimes handles poorly.
3.  **Transparency**: You don't need to do anything. The software automatically detects if the C++ module is available and uses it. If not, it has a failsafe to revert to the old method without crashing.

All code has been updated and integrated into `stacking_suite.py` and `star_alignment.py`.

## Installation & Requirements
To build and run the new C++ engine, you need to ensure your environment is set up correctly.

### 1. Build Requirements
The following tools are required to compile the C++ extension:
- **C++ Compiler**: MSVC (Visual Studio) on Windows, or GCC/Clang on Linux/macOS.
- **CMake**: Build system generator.
- **Ninja**: Fast build tool.

### 2. Python Dependencies
Install the build system dependencies:
```bash
pip install scikit-build-core pybind11 cmake ninja
```

### 3. Building the Extension
If you are running from source:
```bash
pip install -e .
```
This command will automatically compile the `saspro_cpp` module and link it with your Python environment.

