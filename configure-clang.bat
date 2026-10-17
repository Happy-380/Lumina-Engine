@echo off
call "D:\VS2026\VC\Auxiliary\Build\vcvars64.bat" >nul
set PATH=D:\llvm\bin;C:\Program Files\CMake\bin;D:\VS2026\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%
cd /d C:\Lumina-Engine\llama.cpp-b10991
cmake -S . -B build-clang -G Ninja ^
  -DCMAKE_TOOLCHAIN_FILE=cmake/x64-windows-llvm.cmake ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DBUILD_SHARED_LIBS=ON ^
  -DGGML_BACKEND_DL=ON ^
  -DGGML_CPU_ALL_VARIANTS=ON ^
  -DGGML_NATIVE=OFF ^
  -DGGML_RPC=ON ^
  -DGGML_OPENMP=ON ^
  -DLLAMA_BUILD_TESTS=OFF ^
  -DLLAMA_BUILD_EXAMPLES=OFF ^
  -DLLAMA_BUILD_TOOLS=ON ^
  -DLLAMA_BUILD_SERVER=ON ^
  -DLLAMA_BUILD_APP=OFF ^
  -DLLAMA_CURL=OFF ^
  %*
