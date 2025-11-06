@echo off
echo ========================================
echo  Stopping Diarization Service
echo ========================================

:: Ищем процессы Python, которые могут быть сервисом диаризации
echo Looking for running diarization service processes...

:: Показываем все Python процессы для справки
echo.
echo Current Python processes:
wmic process where "name='python.exe'" get ProcessId,CommandLine /format:table 2>nul

:: Попытка найти и остановить процесс с diarization_service.py
for /f "tokens=1" %%i in ('wmic process where "CommandLine like '%%diarization_service.py%%'" get ProcessId /value 2^>nul ^| find "ProcessId"') do (
    set "line=%%i"
    for /f "tokens=2 delims==" %%j in ("!line!") do (
        set "pid=%%j"
        if defined pid if not "!pid!"=="" (
            echo Stopping process ID: !pid!
            taskkill /PID !pid! /F >nul 2>&1
            if !errorlevel! equ 0 (
                echo Successfully stopped process !pid!
            ) else (
                echo Failed to stop process !pid!
            )
        )
    )
)

:: Альтернативный способ - остановка по имени процесса содержащего diarization
echo.
echo Attempting to stop any Python processes running diarization service...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *diarization*" >nul 2>&1

:: Проверяем порт 5000 (по умолчанию)
echo.
echo Checking if port 5000 is still in use...
netstat -an | find ":5000" >nul 2>&1
if %errorlevel% equ 0 (
    echo Port 5000 is still in use by another process
    echo You may need to stop it manually:
    netstat -ano | find ":5000"
) else (
    echo Port 5000 is free
)

echo.
echo ========================================
echo  Stop operation completed
echo ========================================
pause