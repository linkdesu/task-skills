@echo off
REM Build SageAttention 2.2 for AMD RDNA4 (gfx1201) / ROCm 7.2 on Windows.
REM Run from the SageAttention project root (the dir containing .venv and _build_tree).
REM Adjust VCVARS and ROCM_SDK if your layout differs.

set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set "ROCM_SDK=%CD%\.venv\Lib\site-packages\_rocm_sdk_devel"

call "%VCVARS%" >nul 2>&1
if errorlevel 1 (
    echo vcvars64 failed - check VCVARS path in this script
    exit /b 1
)
set "ROCM_HOME=%ROCM_SDK%"
set "PYTORCH_ROCM_ARCH=gfx1201"
set "HIPCC_APPEND_FLAGS=-Wno-invalid-specialization"

uv pip install --python "%CD%\.venv\Scripts\python.exe" --no-build-isolation -v "%CD%\_build_tree"
exit /b %errorlevel%
