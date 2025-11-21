@echo off
echo ========================================
echo  Audio Diarization Service Startup
echo ========================================

:: Переходим в папку сервиса
cd /d %~dp0

:: Проверяем и активируем виртуальное окружение
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
    echo Virtual environment activated
) else if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
    echo Virtual environment activated
) else (
    echo WARNING: Virtual environment not found
    echo Using system Python
)
echo.

:: Проверяем наличие Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Please install Python 3.8+ and add to PATH
    pause
    exit /b 1
)

:: Проверяем наличие основных файлов
if not exist "diarization_service.py" (
    echo ERROR: diarization_service.py not found!
    pause
    exit /b 1
)

if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found!
    pause
    exit /b 1
)

:: Показываем текущую конфигурацию
echo Current directory: %CD%
echo Python version:
python --version
echo.

:: Проверяем .env файл (если есть)
if exist ".env" (
    echo Found .env file - configuration will be loaded from it
) else (
    echo WARNING: No .env file found - using default configuration
    echo You may want to create .env file with:
    echo   HF_TOKEN=your_huggingface_token
    echo   DIARIZATION_PORT=5000
    echo   DIARIZATION_HOST=0.0.0.0
    echo   BEARER_TOKEN=your_secret_token
)
echo.

:: Проверяем установку зависимостей
echo Checking dependencies...
python -c "import flask, pyannote.audio" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing dependencies...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install dependencies!
        echo.
        echo For CUDA support, run setup_cuda.bat instead
        pause
        exit /b 1
    )
) else (
    echo Dependencies OK
)
echo.

:: Проверяем CUDA поддержку
echo Checking GPU/CUDA support...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); cuda_available = torch.cuda.is_available(); print(f'CUDA: {\"Available\" if cuda_available else \"Not available\"}'); print(f'GPU: {torch.cuda.get_device_name(0) if cuda_available else \"None\"}') if cuda_available else None" 2>nul
if %errorlevel% neq 0 (
    echo WARNING: Could not check CUDA status
)
echo.

:: Проверяем наличие модели
if exist "models\pyannote\speaker-diarization-3" (
    echo Local model found in models directory
) else (
    echo WARNING: Local model not found
    echo Service will try to download from HuggingFace on first request
    echo Make sure HF_TOKEN is set if needed
)
echo.

:: Запускаем сервис
echo ========================================
echo  Starting Diarization Service...
echo ========================================
echo.
echo Service will be available at:
echo   - Health check: http://localhost:5000/health
echo   - Diarization: POST http://localhost:5000/diarize
echo.
echo Press Ctrl+C to stop the service
echo ========================================
echo.

python diarization_service.py

:: Если сервис завершился с ошибкой
if %errorlevel% neq 0 (
    echo.
    echo ========================================
    echo Service stopped with error code %errorlevel%
    echo Check the error messages above
    echo ========================================
    pause
)