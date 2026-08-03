@echo off
cd /d C:\WASTE\wastex
set SITE_ROLE=MASTER

:: Activate Master virtual environment (assuming it is called venv_master)
call venv_master\Scripts\activate

:: Run the sync command (Downloads pending images AND automatically uploads the active model)
python manage.py sync_to_cloud >> c:\WASTE\wastex\master_sync.log 2>&1

exit /b 0
