@echo off
REM ════════════════════════════════════════════════════════════════════════════
REM  WasteX Installation Script for Windows
REM  Run this once on each customer site to configure and seed the system.
REM ════════════════════════════════════════════════════════════════════════════

setlocal enabledelayedexpansion

set "GREEN=[32m"
set "YELLOW=[33m"
set "RED=[31m"
set "CYAN=[36m"
set "RESET=[0m"

echo.
echo !GREEN!╔════════════════════════════════════════╗!RESET!
echo !GREEN!║        WasteX Installation Wizard      ║!RESET!
echo !GREEN!║      Waste Classification System       ║!RESET!
echo !GREEN!╚════════════════════════════════════════╝!RESET!
echo.

REM ── 0. Python check ─────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo !RED!Error: Python is not installed or not in PATH.!RESET!
    echo Please install Python 3.9+ from python.org
    pause
    exit /b 1
)
echo !GREEN!✓ Python detected:!RESET!
python --version
echo.

REM ── 1. .env bootstrap ───────────────────────────────────────────────────────
if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo !YELLOW!.env created from .env.example — please review it before continuing.!RESET!
    ) else (
        echo !RED!Error: Neither .env nor .env.example found.!RESET!
        echo Please create a .env file before running this installer.
        pause
        exit /b 1
    )
)

REM ── 2. Generate SECRET_KEY if placeholder is still present ───────────────────
findstr /C:"CHANGE_ME_generate_a_real_key" ".env" >nul 2>&1
if not errorlevel 1 (
    echo !YELLOW!Generating a secure SECRET_KEY...!RESET!
    for /f "tokens=*" %%K in ('python -c "import secrets; print(secrets.token_urlsafe(50))"') do (
        set "NEW_KEY=%%K"
    )
    REM Replace the placeholder line in .env using Python (handles special chars safely)
    python -c "
import re, pathlib
env = pathlib.Path('.env').read_text()
env = re.sub(r'SECRET_KEY=.*', 'SECRET_KEY=!NEW_KEY!', env)
pathlib.Path('.env').write_text(env)
"
    echo !GREEN!✓ SECRET_KEY generated and written to .env!RESET!
)
echo.

REM ── 3. Data storage folder ──────────────────────────────────────────────────
echo !YELLOW!Where should WasteX store media and datasets?!RESET!
echo.
echo  Common options:
echo    D:\WasteX  (separate drive, recommended for large sites)
echo    C:\WasteX  (same drive)
echo.
set /p DATA_PATH="Enter data folder path (default: C:\WasteX): "
if "!DATA_PATH!"=="" set "DATA_PATH=C:\WasteX"
echo.
echo !YELLOW!  Media   : !DATA_PATH!\media!RESET!
echo !YELLOW!  Datasets: !DATA_PATH!\datasets!RESET!
echo.

REM ── 4. DB credentials ───────────────────────────────────────────────────────
echo !YELLOW!PostgreSQL database credentials:!RESET!
set /p DB_NAME="  DB name     (default: wastex):    "
set /p DB_USER="  DB user     (default: postgres):  "
set /p DB_PASSWORD="  DB password (required):          "
set /p DB_HOST="  DB host     (default: localhost): "
set /p DB_PORT="  DB port     (default: 5432):      "

if "!DB_NAME!"==""     set "DB_NAME=wastex"
if "!DB_USER!"==""     set "DB_USER=postgres"
if "!DB_HOST!"==""     set "DB_HOST=localhost"
if "!DB_PORT!"==""     set "DB_PORT=5432"

if "!DB_PASSWORD!"=="" (
    echo !RED!Error: DB_PASSWORD cannot be empty.!RESET!
    pause
    exit /b 1
)
echo.

REM ── 5. LAN IP for ALLOWED_HOSTS ─────────────────────────────────────────────
echo !YELLOW!What is this machine's LAN IP address?!RESET!
echo  (Raspberry Pis need it to reach the server — leave blank to skip for now)
set /p LAN_IP="  LAN IP (e.g. 192.168.1.10): "
echo.

REM ── 6. User account passwords ───────────────────────────────────────────────
echo !YELLOW!Set passwords for the two WasteX accounts:!RESET!
echo.
echo  !CYAN!edge!RESET!   — used daily by the factory/site operator
echo  !CYAN!master!RESET! — used by WasteX company for retraining ^& model management
echo.
set /p EDGE_PASS="  Password for 'edge':   "
set /p MASTER_PASS="  Password for 'master': "
echo.

if "!EDGE_PASS!"=="" (
    echo !RED!Error: edge password cannot be empty.!RESET!
    pause
    exit /b 1
)
if "!MASTER_PASS!"=="" (
    echo !RED!Error: master password cannot be empty.!RESET!
    pause
    exit /b 1
)

REM ── 7. Confirmation ─────────────────────────────────────────────────────────
echo !YELLOW!Configuration Summary!RESET!
echo  ─────────────────────────────────────────────────────────
echo  App directory : %CD%
echo  Data directory: !DATA_PATH!
echo  Database      : !DB_NAME! on !DB_HOST!:!DB_PORT! (user: !DB_USER!)
if not "!LAN_IP!"=="" (
    echo  ALLOWED_HOSTS : localhost,127.0.0.1,!LAN_IP!
) else (
    echo  ALLOWED_HOSTS : localhost,127.0.0.1
)
echo  Accounts      : edge, master
echo  ─────────────────────────────────────────────────────────
echo.
set /p CONFIRM="Proceed with installation? (Y/N): "
if /i not "!CONFIRM!"=="Y" (
    echo Installation cancelled.
    exit /b 0
)
echo.

REM ── 8. Write final .env ─────────────────────────────────────────────────────
echo !YELLOW!Writing configuration to .env...!RESET!

if not "!LAN_IP!"=="" (
    set "ALLOWED_HOSTS_VALUE=localhost,127.0.0.1,!LAN_IP!"
) else (
    set "ALLOWED_HOSTS_VALUE=localhost,127.0.0.1"
)

python -c "
import pathlib, re

env_path = pathlib.Path('.env')
env = env_path.read_text()

replacements = {
    'DEBUG':            'False',
    'ALLOWED_HOSTS':    '!ALLOWED_HOSTS_VALUE!',
    'DB_NAME':          '!DB_NAME!',
    'DB_USER':          '!DB_USER!',
    'DB_PASSWORD':      '!DB_PASSWORD!',
    'DB_HOST':          '!DB_HOST!',
    'DB_PORT':          '!DB_PORT!',
    'WASTE_MEDIA_ROOT':    r'!DATA_PATH!\media',
    'WASTE_DATASETS_ROOT': r'!DATA_PATH!\datasets',
}

for key, value in replacements.items():
    pattern = rf'^{key}=.*'
    replacement = f'{key}={value}'
    if re.search(pattern, env, flags=re.MULTILINE):
        env = re.sub(pattern, replacement, env, flags=re.MULTILINE)
    else:
        env += f'\n{key}={value}'

env_path.write_text(env)
"
echo !GREEN!✓ .env updated!RESET!
echo.

REM ── 9. Install Python dependencies ──────────────────────────────────────────
echo !YELLOW!Installing Python dependencies...!RESET!
pip install -q -r requirements.txt
if errorlevel 1 (
    echo !RED!Error: Failed to install dependencies.!RESET!
    pause
    exit /b 1
)
echo !GREEN!✓ Dependencies installed!RESET!
echo.

REM ── 10. Database migrations ──────────────────────────────────────────────────
echo !YELLOW!Running database migrations...!RESET!
python manage.py migrate --noinput
if errorlevel 1 (
    echo !RED!Error: Database migration failed.!RESET!
    echo Check that PostgreSQL is running and the DB credentials in .env are correct.
    pause
    exit /b 1
)
echo !GREEN!✓ Database ready!RESET!
echo.

REM ── 11. Storage paths ────────────────────────────────────────────────────────
echo !YELLOW!Configuring storage paths...!RESET!
python manage.py initialize_paths ^
    --media-root    "!DATA_PATH!\media"    ^
    --datasets-root "!DATA_PATH!\datasets" ^
    --models-root   "!DATA_PATH!\models"
if errorlevel 1 (
    echo !RED!Error: Storage path initialization failed.!RESET!
    pause
    exit /b 1
)
echo.

REM ── 12. Seed user accounts ───────────────────────────────────────────────────
echo !YELLOW!Creating user accounts...!RESET!
python manage.py seed_installation ^
    --edge-password   "!EDGE_PASS!"   ^
    --master-password "!MASTER_PASS!"
if errorlevel 1 (
    echo !RED!Error: User seeding failed.!RESET!
    pause
    exit /b 1
)

REM ── Done ─────────────────────────────────────────────────────────────────────
echo !GREEN!╔════════════════════════════════════════╗!RESET!
echo !GREEN!║    WasteX Installation Complete! ✅    ║!RESET!
echo !GREEN!╚════════════════════════════════════════╝!RESET!
echo.
echo !YELLOW!Next steps:!RESET!
echo   1. Start the server : python run_server.py
echo   2. Open in browser  : http://localhost:8000
echo   3. Log in as 'edge' (daily use) or 'master' (retraining / admin)
echo.

set /p START_SERVER="Start the server now? (Y/N): "
if /i "!START_SERVER!"=="Y" (
    python run_server.py
)

endlocal
