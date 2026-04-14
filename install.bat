@echo off
REM WasteX Installation Script for Windows
REM This script initializes WasteX with user-selected storage paths

setlocal enabledelayedexpansion

REM Colors (using Windows 10+ ANSI escape codes)
set "GREEN=[32m"
set "YELLOW=[33m"
set "RED=[31m"
set "RESET=[0m"

echo.
echo !GREEN!╔════════════════════════════════════════╗!RESET!
echo !GREEN!║     WasteX Installation Wizard         ║!RESET!
echo !GREEN!║   Waste Classification System         ║!RESET!
echo !GREEN!╚════════════════════════════════════════╝!RESET!
echo.

REM Check Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo !RED!Error: Python is not installed or not in PATH!RESET!
    echo Please install Python 3.9+ from python.org
    pause
    exit /b 1
)

echo !GREEN!✓ Python detected!RESET!
python --version
echo.

REM Ask user for data folder
echo !YELLOW!Where should WasteX store data?!RESET!
echo.
echo Common options:
echo   • D:\WasteX (separate drive, recommended)
echo   • C:\Users\USERNAME\AppData\Local\WasteX (user profile)
echo.
set /p DATA_PATH="Enter data folder path (default: C:\WasteX): "

if "!DATA_PATH!"=="" (
    set "DATA_PATH=C:\WasteX"
)

echo.
echo !YELLOW!Configuration Summary:!RESET!
echo   Installation Directory: %CD%
echo   Data Directory: !DATA_PATH!
echo     ├─ Media (images): !DATA_PATH!\media
echo     ├─ Datasets (training): !DATA_PATH!\datasets
echo     └─ Models (ML models): !DATA_PATH!\models
echo.

set /p CONFIRM="Is this correct? (Y/N): "
if /i not "!CONFIRM!"=="Y" (
    echo Installation cancelled.
    exit /b 0
)

echo.
echo !YELLOW!Installing dependencies...!RESET!
pip install -q -r requirements.txt
if errorlevel 1 (
    echo !RED!Error: Failed to install dependencies!RESET!
    pause
    exit /b 1
)
echo !GREEN!✓ Dependencies installed!RESET!

echo.
echo !YELLOW!Initializing database...!RESET!
python manage.py migrate --noinput
if errorlevel 1 (
    echo !RED!Error: Database migration failed!RESET!
    pause
    exit /b 1
)
echo !GREEN!✓ Database initialized!RESET!

echo.
echo !YELLOW!Configuring storage paths...!RESET!
python manage.py initialize_paths ^
    --media-root "!DATA_PATH!\media" ^
    --datasets-root "!DATA_PATH!\datasets" ^
    --models-root "!DATA_PATH!\models"

if errorlevel 1 (
    echo !RED!Error: Path initialization failed!RESET!
    pause
    exit /b 1
)

echo.
echo !GREEN!╔════════════════════════════════════════╗!RESET!
echo !GREEN!║   Installation Completed Successfully! ║!RESET!
echo !GREEN!╚════════════════════════════════════════╝!RESET!
echo.
echo !YELLOW!Next steps:!RESET!
echo   1. Start the server: python manage.py runserver
echo   2. Open browser: http://localhost:8000
echo   3. You're ready to classify waste!
echo.
echo !YELLOW!Data will be stored at: !DATA_PATH!!RESET!
echo.

set /p START_SERVER="Start server now? (Y/N): "
if /i "!START_SERVER!"=="Y" (
    python manage.py runserver
) else (
    echo To start manually, run:
    echo   python manage.py runserver
)

endlocal
