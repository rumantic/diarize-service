@echo off
echo ========================================
echo  Diarization Service - First Time Setup
echo ========================================

:: Переходим в папку сервиса
cd /d %~dp0

:: Проверяем Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found! 
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to add Python to PATH during installation
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

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

:: Устанавливаем зависимости
echo Installing Python dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies!
    pause
    exit /b 1
)

echo Dependencies installed successfully!
echo.

:: Создаем .env файл если его нет
if not exist ".env" (
    echo Creating .env configuration file...
    (
        echo # Diarization Service Configuration
        echo.
        echo # HuggingFace token for downloading models
        echo # Get token from https://huggingface.co/settings/tokens
        echo # Accept license at https://huggingface.co/pyannote/speaker-diarization-3.1
        echo HF_TOKEN=your_huggingface_token_here
        echo.
        echo # Service configuration
        echo DIARIZATION_PORT=5000
        echo DIARIZATION_HOST=0.0.0.0
        echo.
        echo # API security token
        echo BEARER_TOKEN=your_secret_bearer_token_here
        echo.
        echo # Optional: Proxy settings (if needed)
        echo #HTTP_PROXY=http://proxy.company.com:8080
        echo #HTTPS_PROXY=http://proxy.company.com:8080
        echo #NO_PROXY=localhost,127.0.0.1
    ) > .env
    
    echo .env file created!
    echo.
    echo IMPORTANT: Edit .env file and set:
    echo   1. HF_TOKEN - your HuggingFace token
    echo   2. BEARER_TOKEN - your API security token
    echo.
) else (
    echo .env file already exists
)

:: Проверяем наличие модели
if exist "models\pyannote\speaker-diarization-3" (
    echo Local model found - setup complete!
) else (
    echo.
    echo ========================================
    echo  Model Download Required
    echo ========================================
    echo.
    echo To download the model:
    echo   1. Edit .env file and set your HF_TOKEN
    echo   2. Run: python download_model.py
    echo.
    echo Or the model will be downloaded automatically on first use
    echo (requires HF_TOKEN to be set)
)

echo.
echo ========================================
echo  Setup Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Edit .env file with your tokens
echo   2. Run start_service.bat to start the service
echo   3. Test with: curl -X GET http://localhost:5000/health
echo.
pause