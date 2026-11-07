@echo off
call "D:\VS2026\VC\Auxiliary\Build\vcvars64.bat" >nul
REM Vulkan backend: the SDK is not admin-installed on this machine; it lives in D:\VulkanSDK\sdk
REM (extracted from the official 1.4.363 installer). Both trees autodetect it the same way.
if exist "D:\VulkanSDK\sdk\Bin\glslc.exe" (
  set "VULKAN_SDK=D:\VulkanSDK\sdk"
  set "CMAKE_PREFIX_PATH=D:/VulkanSDK/sdk"
  set "GGML_VULKAN_OPT=-DGGML_VULKAN=ON"
) else (
  echo [configure-clang] warning: Vulkan SDK not found, building without Vulkan
  set "GGML_VULKAN_OPT=-DGGML_VULKAN=OFF"
)
set PATH=D:\llvm\bin;C:\Program Files\CMake\bin;D:\VS2026\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%
if defined VULKAN_SDK set PATH=%VULKAN_SDK%\Bin;%PATH%
cd /d "%~dp0llama.cpp-b10991"
REM Binaries are emitted straight into the delivery directory (no <build>/bin copy).
REM The MSVC tree keeps its own bin\ because its 9 CPU variants are the ones that ship.
set "LUMINA_BIN=%~dp0Lumina-Engine-Bin"
REM forward slashes: the ggml-vulkan host toolchain file does `set(X ${value})` unquoted,
REM a backslash path would be parsed as a broken escape sequence
set "LUMINA_BIN=%LUMINA_BIN:\=/%"
cmake -S . -B build-clang -G Ninja ^
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=%LUMINA_BIN% ^
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=%LUMINA_BIN% ^
  -DCMAKE_TOOLCHAIN_FILE=cmake/x64-windows-llvm.cmake ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DBUILD_SHARED_LIBS=ON ^
  -DGGML_BACKEND_DL=ON ^
  -DGGML_CPU_ALL_VARIANTS=ON ^
  -DGGML_NATIVE=OFF ^
  -DGGML_RPC=ON ^
  -DGGML_OPENMP=ON ^
  %GGML_VULKAN_OPT% ^
  -DLLAMA_BUILD_TESTS=OFF ^
  -DLLAMA_BUILD_EXAMPLES=OFF ^
  -DLLAMA_BUILD_TOOLS=ON ^
  -DLLAMA_BUILD_SERVER=ON ^
  -DLLAMA_BUILD_APP=OFF ^
  -DLLAMA_CURL=OFF ^
  %*
