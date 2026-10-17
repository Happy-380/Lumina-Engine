@echo off
call "D:\VS2026\VC\Auxiliary\Build\vcvars64.bat" >nul
set PATH=C:\Program Files\CMake\bin;D:\VS2026\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%
cd /d C:\Lumina-Engine\llama.cpp-b10991
cmake -S . -B build -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DLLAMA_BUILD_TESTS=OFF ^
  -DLLAMA_BUILD_EXAMPLES=OFF ^
  -DLLAMA_BUILD_SERVER=OFF ^
  -DLLAMA_BUILD_APP=OFF ^
  -DGGML_NATIVE=OFF ^
  -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON ^
  %*
