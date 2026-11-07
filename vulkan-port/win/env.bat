@echo off
REM Lumina-Engine Vulkan port - Windows build environment
REM Adjust these paths if your machine differs.

set "LUMINA_ROOT=%~dp0..\..\llama.cpp-b10991"
set "LUMINA_BUILD=%~dp0..\..\llama.cpp-b10991\build-vulkan"

set "VS_VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "VULKAN_SDK=D:\VulkanSDK\sdk"
set "CMAKE_BIN=C:\Program Files\CMake\bin"

if not exist "%VS_VCVARS%" (
  echo [env] vcvars64.bat not found at "%VS_VCVARS%"
  exit /b 1
)
if not exist "%VULKAN_SDK%\Bin\glslc.exe" (
  echo [env] Vulkan SDK glslc not found at "%VULKAN_SDK%\Bin\glslc.exe"
  exit /b 1
)

call "%VS_VCVARS%" >nul
set "PATH=%CMAKE_BIN%;%VULKAN_SDK%\Bin;%PATH%"
exit /b 0
