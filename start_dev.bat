@echo off
echo ========================================
echo  Diarization Service - Development Mode
echo ========================================

:: Переходим в папку сервиса
cd /d %~dp0

:: Активируем виртуальное окружение если есть
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

:: Устанавливаем переменные окружения для разработки
set FLASK_ENV=development
set FLASK_DEBUG=1

:: Переменные для более подробного логирования
set PYTHONUNBUFFERED=1

echo Development mode enabled
echo - Flask debug mode: ON
echo - Detailed logging: ON
echo.

:: Запускаем основной скрипт
call start_service.bat