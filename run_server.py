#!/usr/bin/env python
"""Run Django development server"""
import os
import sys
import django
from django.core.management import execute_from_command_line

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'wastex.settings')
django.setup()

if __name__ == '__main__':
    print("DEBUG: About to start server", flush=True)
    sys.stdout.flush()
    try:
        execute_from_command_line([
            'manage.py',
            'runserver',
            '0.0.0.0:8000',
            '--nothreading',
            '--noreload'
        ])
    except Exception as e:
        print(f"DEBUG: Server crashed with exception: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
