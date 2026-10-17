@echo off
call "D:\VS2026\VC\Auxiliary\Build\vcvars64.bat" >nul
set PATH=C:\Program Files\CMake\bin;D:\VS2026\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%
REM Vulkan backend: the LunarG SDK was unpacked to D:\VulkanSDK\sdk (no admin install).
if exist "D:\VulkanSDK\sdk\Bin\glslc.exe" (
  set "VULKAN_SDK=D:\VulkanSDK\sdk"
  set "CMAKE_PREFIX_PATH=D:/VulkanSDK/sdk"
  set "GGML_VULKAN_OPT=-DGGML_VULKAN=ON"
) else (
  echo [configure-release] warning: Vulkan SDK not found, building without Vulkan
  set "GGML_VULKAN_OPT=-DGGML_VULKAN=OFF"
)
cd /d C:\Lumina-Engine\llama.cpp-b10991
cmake -S . -B build-release -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DBUILD_SHARED_LIBS=ON ^
  -DGGML_BACKEND_DL=ON ^
  -DGGML_CPU_ALL_VARIANTS=ON ^
  -DGGML_NATIVE=OFF ^
  -DGGML_RPC=ON ^
  %GGML_VULKAN_OPT% ^
  -DLLAMA_BUILD_TESTS=OFF ^
  -DLLAMA_BUILD_EXAMPLES=OFF ^
  -DLLAMA_BUILD_TOOLS=ON ^
  -DLLAMA_BUILD_SERVER=ON ^
  -DLLAMA_BUILD_APP=OFF ^
  -DLLAMA_CURL=OFF ^
  %*
