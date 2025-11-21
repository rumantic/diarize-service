@echo off
echo ========================================
echo  Setup CUDA Support for Diarization Service
echo ========================================

cd /d %~dp0

:: Создаем виртуальное окружение если его нет
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo Virtual environment created
) else (
    echo Virtual environment already exists
)
echo.

:: Активируем виртуальное окружение
echo Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment!
    pause
    exit /b 1
)
echo.

echo Checking CUDA availability...
echo.

:: Проверяем наличие nvidia-smi
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: NVIDIA drivers not found or nvidia-smi not available
    echo Please install NVIDIA GPU drivers first
    echo Download from: https://www.nvidia.com/drivers/
    pause
    exit /b 1
)

echo NVIDIA GPU detected:
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
echo.

:: Проверяем CUDA версию
echo Checking CUDA version...
nvcc --version 2>nul | find "release" 
if %errorlevel% neq 0 (
    echo WARNING: CUDA toolkit not found
    echo PyTorch will try to use bundled CUDA libraries
) else (
    echo CUDA toolkit found
)
echo.

:: Определяем версию CUDA для PyTorch
set CUDA_VERSION=cu121
echo Default CUDA version for PyTorch: %CUDA_VERSION%
echo.

echo Available CUDA versions for PyTorch:
echo   cu118 - CUDA 11.8
echo   cu121 - CUDA 12.1
echo   cpu   - CPU only (no CUDA)
echo.

set /p USER_CUDA="Enter CUDA version (cu118/cu121/cpu) or press Enter for default [%CUDA_VERSION%]: "
if not "%USER_CUDA%"=="" set CUDA_VERSION=%USER_CUDA%

echo.
echo Selected CUDA version: %CUDA_VERSION%
echo.

:: Устанавливаем PyTorch с CUDA
if "%CUDA_VERSION%"=="cpu" (
    echo Installing CPU-only PyTorch...
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
) else if "%CUDA_VERSION%"=="cu118" (
    echo Installing PyTorch with CUDA 11.8...
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
) else if "%CUDA_VERSION%"=="cu121" (
    echo Installing PyTorch with CUDA 12.1...
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo ERROR: Invalid CUDA version: %CUDA_VERSION%
    pause
    exit /b 1
)

if %errorlevel% neq 0 (
    echo ERROR: Failed to install PyTorch with CUDA support
    pause
    exit /b 1
)

echo.
echo Installing other dependencies...
pip install pyannote.audio==3.1.1 soundfile numpy^<2.0.0 huggingface_hub^<0.20.0 hf-transfer^>=0.1.0
pip install flask^>=3.0.0 werkzeug^>=3.0.0 python-dotenv^>=1.0.0 requests^>=2.31.0

echo.
echo ========================================
echo  Testing CUDA Support
echo ========================================

python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else 'CUDA: Not available'); print(f'Device count: {torch.cuda.device_count()}' if torch.cuda.is_available() else ''); [print(f'Device {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None"

echo.
if %errorlevel% equ 0 (
    echo ✓ CUDA setup completed successfully!
    echo.
    echo Next steps:
    echo   1. Run 'start_service.bat' to start the service
    echo   2. Check 'http://localhost:5000/system' for detailed GPU info
    echo   3. The service will automatically use GPU if available
) else (
    echo ✗ CUDA setup failed. Check error messages above.
)

echo.
pause