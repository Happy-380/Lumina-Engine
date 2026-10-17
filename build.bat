@echo off
call "D:\VS2026\VC\Auxiliary\Build\vcvars64.bat" >nul
set PATH=C:\Program Files\CMake\bin;D:\VS2026\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%
cd /d C:\Lumina-Engine\llama.cpp-b10991
cmake --build build --target %* -j 8
