@echo off
REM WasteX GUI Installation Wrapper
REM Launches the Python GUI installer

setlocal enabledelayedexpansion

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0

REM Try to use Python 3.14 first (has tkinter), fall back to system Python
set PYTHON_314=C:\Python314\python.exe
set PYTHON_CMD=python

if exist "%PYTHON_314%" (
    set PYTHON_CMD=%PYTHON_314%
)

REM Check if Python is installed
%PYTHON_CMD% --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ╔════════════════════════════════════════╗
    echo ║  ERROR: Python not found!              ║
    echo ╚════════════════════════════════════════╝
    echo.
    echo Python 3.10 or higher is required.
    echo Please install Python from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

REM Test tkinter
%PYTHON_CMD% -c "import tkinter" >nul 2>&1
if errorlevel 1 (
    echo.
    echo ╔════════════════════════════════════════╗
    echo ║  ERROR: tkinter not found!             ║
    echo ╚════════════════════════════════════════╝
    echo.
    echo tkinter is required but not installed.
    echo Please reinstall Python with tkinter enabled.
    echo.
    pause
    exit /b 1
)

REM Launch the GUI installer
echo Starting WasteX Installation Wizard...
echo.

%PYTHON_CMD% "%SCRIPT_DIR%installer_gui.py"

exit /b %errorlevel%
