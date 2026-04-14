@echo off
REM WasteX GUI Installer Test Script
REM This script tests the GUI installer on your laptop

echo.
echo ╔════════════════════════════════════════════════════════╗
echo ║  WasteX GUI Installer - Test Launcher                 ║
echo ╚════════════════════════════════════════════════════════╝
echo.

REM Try to use Python 3.14 first (has tkinter), fall back to system Python
set PYTHON_314=C:\Python314\python.exe
set PYTHON=python

if exist "%PYTHON_314%" (
    set PYTHON=%PYTHON_314%
)

REM Verify Python is available
%PYTHON% --version >nul 2>&1
if errorlevel 1 (
    echo ╔════════════════════════════════════════════════════════╗
    echo ║  ERROR: Python not found!                             ║
    echo ╚════════════════════════════════════════════════════════╝
    echo.
    echo Python is required to run the installer GUI.
    echo Please install Python from https://www.python.org/
    echo.
    pause
    exit /b 1
)

echo Python found: %PYTHON%
echo.

REM Test tkinter availability
echo Testing tkinter availability...
%PYTHON% -c "import tkinter; print('✓ tkinter is available')" >nul 2>&1

if errorlevel 1 (
    echo ╔════════════════════════════════════════════════════════╗
    echo ║  ERROR: tkinter not available!                        ║
    echo ╚════════════════════════════════════════════════════════╝
    echo.
    echo tkinter is required but not found.
    echo It usually comes with Python by default.
    echo Please reinstall Python.
    echo.
    pause
    exit /b 1
)

echo ✓ tkinter is available
echo.

REM Launch the GUI installer
echo Launching WasteX GUI Installer...
echo.

%PYTHON% "%~dp0installer_gui.py"

if errorlevel 1 (
    echo.
    echo ╔════════════════════════════════════════════════════════╗
    echo ║  ERROR: Installer failed!                             ║
    echo ╚════════════════════════════════════════════════════════╝
    echo.
    pause
    exit /b 1
)

echo.
echo ╔════════════════════════════════════════════════════════╗
echo ║  Installer completed successfully!                    ║
echo ╚════════════════════════════════════════════════════════╝
echo.

exit /b 0
