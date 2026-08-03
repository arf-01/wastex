@echo off
cd /d C:\WASTE\wastex
set SITE_ROLE=EDGE
call venv_edge\Scripts\activate

:: Run the sync command (Uploads images AND downloads models automatically)
python manage.py sync_to_cloud >> c:\WASTE\wastex\edge_sync.log 2>&1

exit /b 0
