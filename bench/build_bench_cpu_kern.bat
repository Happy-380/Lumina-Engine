@echo off
REM ---------------------------------------------------------------------------
REM Build the standalone CPU kernel micro-benchmark used to A/B test the repacked
REM mat-mul kernels (see tests/bench-cpu-kern.cpp and gen_q4k_body.py).
REM
REM It configures a *separate* build tree (build-bench) so that the delivery trees
REM (build-release / build-clang) are not touched. The tree is built with a fixed
REM haswell feature set so that the library kernels and the benchmark copy of the
REM kernel are compiled with identical flags (otherwise the comparison is biased
REM by -march/-mtune differences).
REM
REM usage: bench\build_bench_cpu_kern.bat
REM ---------------------------------------------------------------------------
setlocal EnableExtensions
set "ROOT=%~dp0.."
set "SRC=%ROOT%\llama.cpp-b10991"

call "D:\VS2026\VC\Auxiliary\Build\vcvars64.bat" >nul || goto :no_vs
set "PATH=D:\llvm\bin;C:\Program Files\CMake\bin;D:\VS2026\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
cd /d "%SRC%" || goto :no_src

REM regenerate the baseline kernel body (must run *before* modifying repack.cpp;
REM tests\q4k_gemv_body.inc is the pre-optimisation baseline used for A/B runs)
python "%ROOT%\bench\gen_q4k_body.py" > "tests\q4k_gemv_body.inc" || goto :no_python

cmake -S . -B build-bench -G Ninja ^
  -DCMAKE_TOOLCHAIN_FILE=cmake/x64-windows-llvm.cmake ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DBUILD_SHARED_LIBS=OFF ^
  -DGGML_BACKEND_DL=OFF ^
  -DGGML_CPU_ALL_VARIANTS=OFF ^
  -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON -DGGML_SSE42=ON ^
  -DGGML_RPC=OFF -DGGML_OPENMP=OFF ^
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF ^
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_CURL=OFF || goto :fail

cmake --build build-bench --target bench-cpu-kern -j 4 || goto :fail

echo.
echo [build_bench_cpu_kern] done: %SRC%\build-bench\bin\bench-cpu-kern.exe
echo   correctness only : bench-cpu-kern.exe --n 1024 --nc 4096 --check-only
echo   DRAM sized A/B   : bench-cpu-kern.exe --n 2048 --nc 16384 --iters 120 --rounds 24
echo   L3 sized A/B     : bench-cpu-kern.exe --n 1024 --nc 1024 --iters 2000 --rounds 14
exit /b 0

:no_vs
echo [build_bench_cpu_kern] vcvars64.bat not found, edit the path in this script
exit /b 1
:no_src
echo [build_bench_cpu_kern] missing source tree "%SRC%"
exit /b 1
:no_python
echo [build_bench_cpu_kern] python not available, cannot regenerate q4k_gemv_body.inc
exit /b 1
:fail
echo [build_bench_cpu_kern] build failed
exit /b 1
