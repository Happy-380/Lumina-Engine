@echo off
REM ---------------------------------------------------------------------------
REM Assemble the Lumina-Engine runtime into the default delivery directory:
REM     <repo root>\Lumina-Engine-Bin      (i.e. C:\Lumina-Engine\Lumina-Engine-Bin)
REM
REM Layout: when a clang build is available it is used as the base (tools + all
REM 14 ggml-cpu variants) and the 9 MSVC variants overwrite their clang
REM counterparts. When clang is not installed, the MSVC build is used as the base
REM and the 5 clang-only CPU variants are restored from bench\clang_variants_backup
REM so the delivery still ships the full 14-variant set. The five runtime
REM redistributables are copied from the system.
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

set "SRC_BASE=%SRC_CLANG%"
set "HAVE_CLANG=1"
if not exist "%SRC_CLANG%" (
    echo [assemble_bin] warning: clang build not found, using the MSVC build as base
    set "SRC_BASE=%SRC_MSVC%"
    set "HAVE_CLANG=0"
)

if exist "%DST%" rd /s /q "%DST%"
mkdir "%DST%"

xcopy /y /q /i "%SRC_BASE%\*" "%DST%\" >nul

for %%v in (x64 sse42 sandybridge haswell skylakex cannonlake cascadelake icelake alderlake) do (
    copy /y "%SRC_MSVC%\ggml-cpu-%%v.dll" "%DST%\" >nul
)

if "%HAVE_CLANG%"=="0" (
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

set /a COUNT=0
for %%f in ("%DST%\*") do set /a COUNT+=1
echo [assemble_bin] wrote %COUNT% files to "%DST%"
endlocal
