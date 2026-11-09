@echo off
rem strata-dsv41.exe (the DeepSeek-V4.1 engine) for NVIDIA on Windows, built next to the release engine so package_engine.py ships it
rem beside strata.exe (setup.py dsv41_exe() looks for it in the engine folder).  cudart is static; the only extra DLL is cublas64_13.dll,
rem which package_engine.py copies from CUDA_PATH like it does for strata.exe.
rem   build_dsv41_windows.bat <build dir> [source dir]        env: STRATA_GGML_DIR (optional override: a llama.cpp checkout, kept for two releases; default the vendored third_party/strata-ggml), CUDA_ROOT (default CUDA 13.0), BUILD_JOBS (default 4),
rem                                                            DSV41_ARCH (default "86-real;89-real;120" = sm_86 + sm_89 + sm_120 SASS, PTX sm_120)
rem Result: <build dir>\strata-dsv41.exe (about 66 MB).  Windows: only the cuda13 variant gets it (sm_120 needs CUDA 13).
setlocal
set "BD=%~1"
set "SRCD=%~2"
if "%BD%"=="" ( echo usage: build_dsv41_windows.bat ^<build dir^> [source dir] & exit /b 2 )
if "%SRCD%"=="" set "SRCD=%~dp0.."
if "%CUDA_ROOT%"=="" set "CUDA_ROOT=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0"
if "%DSV41_ARCH%"=="" set "DSV41_ARCH=86-real;89-real;120"
if "%BUILD_JOBS%"=="" set "BUILD_JOBS=4"
set "GGML="
if defined STRATA_GGML_DIR set "GGML=-DSTRATA_GGML_DIR=%STRATA_GGML_DIR:\=/%"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
rem configure every time: cmake bakes the build id (the commit) in at configure, so a build dir left from an older commit would print the old one
if exist "%SRCD%\CMakeLists.txt" (
  cmake -G Ninja -S "%SRCD%" -B "%BD%" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_DSV41=ON -DSTRATA_PORTABLE=ON "-DCMAKE_CUDA_ARCHITECTURES=%DSV41_ARCH%" "-DCMAKE_CUDA_COMPILER=%CUDA_ROOT%/bin/nvcc.exe" "-DCUDAToolkit_ROOT=%CUDA_ROOT%" %GGML%
  if errorlevel 1 exit /b 1
)
rem a stale exe must not pass for this build, and the build id is baked in when cmake configures (a git checkout gives it; a tarball needs BUILD_ID)
if exist "%BD%\strata-dsv41.exe" del /q "%BD%\strata-dsv41.exe"
start "" /belownormal /wait /b cmake --build "%BD%" -j %BUILD_JOBS% --target strata-dsv41
if errorlevel 1 exit /b 1
if not exist "%BD%\strata-dsv41.exe" ( echo no strata-dsv41.exe & exit /b 1 )
for /f "delims=" %%V in ('"%BD%\strata-dsv41.exe" --version') do set "DSV41_VERSION=%%V"
echo %DSV41_VERSION%
if not "%DSV41_VERSION:build unknown=%"=="%DSV41_VERSION%" if not "%ALLOW_UNKNOWN_BUILD_ID%"=="1" ( echo strata-dsv41.exe prints no commit: build from a git checkout or put the commit in BUILD_ID & exit /b 1 )
echo built %BD%\strata-dsv41.exe
exit /b 0
