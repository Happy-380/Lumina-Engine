@echo off
REM ---------------------------------------------------------------------------
REM Build both trees (MSVC + clang) and assemble the runtime into the default
REM delivery directory C:\Lumina-Engine\Lumina-Engine-Bin via bench\assemble_bin.bat.
REM ---------------------------------------------------------------------------
call "%~dp0bench\build_all_msvc.bat"
if errorlevel 1 ( echo [build-bin] MSVC build failed & exit /b 1 )

REM The clang build produces 5 extra CPU variants (ivybridge, piledriver,
REM cooperlake, zen4, sapphirerapids). It is optional: when clang is not
REM installed, assemble_bin.bat falls back to the MSVC base and restores those
REM variants from bench\clang_variants_backup.
where clang >nul 2>nul
if errorlevel 1 (
    echo [build-bin] warning: clang not found in PATH, skipping the clang build
) else (
    call "%~dp0bench\build_all_clang.bat"
    if errorlevel 1 ( echo [build-bin] clang build failed & exit /b 1 )
)

call "%~dp0bench\assemble_bin.bat"
if errorlevel 1 ( echo [build-bin] assembly failed & exit /b 1 )

echo [build-bin] done
