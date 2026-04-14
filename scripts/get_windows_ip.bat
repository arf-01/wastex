@echo off
REM Find your Windows machine's IP address
REM This is what you need to put in the Pi's BACKEND_URL

echo.
echo ==========================================
echo Finding Your Windows Machine's IP Address
echo ==========================================
echo.

ipconfig /all | findstr /I "IPv4 Address"

echo.
echo ==========================================
echo Look for your ACTIVE connection's IPv4 address (192.168.x.x)
echo Use this IP in the Pi's BACKEND_URL setting!
echo Example: BACKEND_URL = "http://192.168.1.50:8000"
echo ==========================================
echo.
pause
