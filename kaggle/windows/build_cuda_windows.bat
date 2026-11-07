@echo off
REM ---------------------------------------------------------------------------
REM Build a CUDA-enabled Lumina-Engine on Windows (optional).
REM
REM The Kaggle build (kaggle\build_cuda.sh) produces Linux binaries; this script
REM is the local equivalent if you later want to run the CUDA build on a Windows
REM machine with an NVIDIA GPU.
REM
REM Requirements:
REM   * Visual Studio 2022/2026 with the C++ workload
REM   * NVIDIA CUDA Toolkit (nvcc on PATH)          -> https://developer.nvidia.com/cuda-downloads
REM   * CMake + Ninja (the script looks in the usual places)
REM
REM Output: D:\Lumina-Engine\kaggle\bin-windows-cuda\Lumina-Engine-train.exe
REM         D:\Lumina-Engine\kaggle\bin-windows-cuda\Lumina-Engine-cli.exe
REM
REM NOTE: training still runs on the CPU (the backward pass has no CUDA kernels);
REM       the CUDA build is for inference and for the experimental
REM       LUMINA_TRAIN_ALLOW_GPU=1 path. See kaggle\README.md section 4.
REM ---------------------------------------------------------------------------
setlocal EnableExtensions
set "ROOT=%~dp0..\.."
set "SRC=%ROOT%\llama.cpp-b10991"
set "BUILD=%ROOT%\kaggle\build-windows-cuda"
set "DST=%ROOT%\kaggle\bin-windows-cuda"

if not exist "%SRC%\CMakeLists.txt" (
    echo [win-cuda] source tree not found: "%SRC%"
    exit /b 1
)

REM --- Visual Studio environment ---------------------------------------------
set "VS_FOUND="
for %%d in ("D:\VS2026" "D:\VS2022" "C:\Program Files\Microsoft Visual Studio\2022\Community") do (
    if exist "%%~d\VC\Auxiliary\Build\vcvars64.bat" (
        call "%%~d\VC\Auxiliary\Build\vcvars64.bat" >nul
        set "VS_FOUND=%%~d"
        goto :vs_ok
    )
)
:vs_ok
if not defined VS_FOUND (
    echo [win-cuda] vcvars64.bat not found - run this from a "x64 Native Tools" prompt
    exit /b 1
)
echo [win-cuda] Visual Studio: %VS_FOUND%

set "PATH=C:\Program Files\CMake\bin;%PATH%"
where nvcc >nul 2>nul
if errorlevel 1 (
    echo [win-cuda] nvcc not found - install the NVIDIA CUDA Toolkit first
    exit /b 1
)
for /f "delims=" %%v in ('nvcc --version ^| findstr /r "release"') do echo [win-cuda] %%v

REM --- configure + build ------------------------------------------------------
cmake -S "%SRC%" -B "%BUILD%" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DBUILD_SHARED_LIBS=OFF ^
  -DGGML_NATIVE=ON ^
  -DGGML_CUDA=ON ^
  -DGGML_CUDA_FA=ON ^
  -DGGML_CUDA_NCCL=OFF ^
  -DGGML_BACKEND_DL=OFF ^
  -DGGML_CPU_ALL_VARIANTS=OFF ^
  -DGGML_RPC=OFF ^
  -DLLAMA_CURL=OFF ^
  -DLLAMA_BUILD_TESTS=OFF ^
  -DLLAMA_BUILD_EXAMPLES=OFF ^
  -DLLAMA_BUILD_APP=OFF ^
  -DLLAMA_BUILD_SERVER=ON ^
  -DLLAMA_BUILD_TOOLS=ON ^
  -DLLAMA_BUILD_UI=OFF ^
  -DLLAMA_USE_PREBUILT_UI=OFF
if errorlevel 1 (
    echo [win-cuda] cmake configure failed
    exit /b 1
)

cmake --build "%BUILD%" --target llama-train llama-cli -j 8
if errorlevel 1 (
    echo [win-cuda] build failed
    exit /b 1
)

if not exist "%DST%" mkdir "%DST%"
copy /y "%BUILD%\bin\Lumina-Engine-train.exe" "%DST%\" >nul
copy /y "%BUILD%\bin\Lumina-Engine-cli.exe"   "%DST%\" >nul
echo [win-cuda] done: "%DST%"
"%DST%\Lumina-Engine-train.exe" --version
endlocal
