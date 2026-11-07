@echo off
REM ---------------------------------------------------------------------------
REM Finalise the Lumina-Engine runtime directory <repo root>\Lumina-Engine-Bin.
REM
REM Since Build 31 the Clang build links *directly* into Lumina-Engine-Bin
REM (-DCMAKE_RUNTIME_OUTPUT_DIRECTORY/-DCMAKE_LIBRARY_OUTPUT_DIRECTORY in
REM configure-clang.bat), so there is no duplicated <build-clang>\bin anymore.
REM
REM This script therefore only does the two things a plain build cannot:
REM   1. overlay the 9 MSVC-compiled CPU variants (build-release\bin) on top of
REM      the clang baseline -- on Windows MSVC stays the default toolchain, and
REM      the 5 remaining clang-only variants (ivybridge, piledriver, cooperlake,
REM      zen4, sapphirerapids) cannot be built with MSVC at all;
REM   2. copy the 5 runtime redistributables from System32.
REM
REM Fallback: when no clang build is available (clang not installed / build tree
REM removed) the whole MSVC bin becomes the base and the 5 clang-only variants
REM are restored from bench\clang_variants_backup.
REM ---------------------------------------------------------------------------
setlocal EnableExtensions
set "ROOT=%~dp0.."
set "SRC_CLANG=%ROOT%\llama.cpp-b10991\build-clang\bin"
set "SRC_MSVC=%ROOT%\llama.cpp-b10991\build-release\bin"
set "DST=%ROOT%\Lumina-Engine-Bin"

if not exist "%SRC_MSVC%" (
    echo [assemble_bin] missing msvc build: "%SRC_MSVC%"
    exit /b 1
)
if not exist "%DST%" mkdir "%DST%"

REM Has the clang build already produced the delivery baseline in %DST%?
set "HAVE_CLANG_DST=0"
if exist "%DST%\Lumina-Engine-cli.exe" if exist "%DST%\ggml-base.dll" set "HAVE_CLANG_DST=1"

REM Legacy layout (clang build tree still emitting into build-clang\bin)
if "%HAVE_CLANG_DST%"=="0" if exist "%SRC_CLANG%\Lumina-Engine-cli.exe" (
    echo [assemble_bin] using the legacy staging directory "%SRC_CLANG%"
    xcopy /y /q /i "%SRC_CLANG%\*" "%DST%\" >nul
    set "HAVE_CLANG_DST=1"
)

if "%HAVE_CLANG_DST%"=="0" (
    echo [assemble_bin] warning: no clang build found, using the MSVC build as the base
    xcopy /y /q /i "%SRC_MSVC%\*" "%DST%\" >nul
)

for %%v in (x64 sse42 sandybridge haswell skylakex cannonlake cascadelake icelake alderlake) do (
    copy /y "%SRC_MSVC%\ggml-cpu-%%v.dll" "%DST%\" >nul
)

if "%HAVE_CLANG_DST%"=="0" (
    for %%v in (ivybridge piledriver cooperlake zen4 sapphirerapids) do (
        if exist "%~dp0clang_variants_backup\ggml-cpu-%%v.dll" (
            copy /y "%~dp0clang_variants_backup\ggml-cpu-%%v.dll" "%DST%\" >nul
        ) else (
            echo [assemble_bin] warning: missing clang-only variant ggml-cpu-%%v.dll
        )
    )
)

for %%f in (libomp140.x86_64.dll msvcp140.dll vcomp140.dll vcruntime140.dll vcruntime140_1.dll) do (
    if exist "%SystemRoot%\System32\%%f" (
        copy /y "%SystemRoot%\System32\%%f" "%DST%\" >nul
    ) else (
        echo [assemble_bin] warning: %%f not found in System32
    )
)

REM vulkan-shaders-gen is a build-time host tool (it needs the Vulkan SDK), never ship it
if exist "%DST%\vulkan-shaders-gen.exe" del /q "%DST%\vulkan-shaders-gen.exe"

set /a COUNT=0
for %%f in ("%DST%\*") do set /a COUNT+=1
echo [assemble_bin] wrote %COUNT% files to "%DST%"
endlocal
