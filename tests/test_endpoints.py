import requests
import time

# Give the server a moment to fully start
time.sleep(2)

urls_to_test = [
    'http://127.0.0.1:8000/classifier/dashboard/',
    'http://127.0.0.1:8000/classifier/api/counts/',
    'http://127.0.0.1:8000/classifier/api/ood/',
]

print("Testing Django endpoints...")
for url in urls_to_test:
    try:
        response = requests.get(url, timeout=5)
        status = "✓" if response.status_code == 200 else "✗"
        print(f"{status} {url}: {response.status_code}")
    except Exception as e:
        print(f"✗ {url}: ERROR - {e}")

print("\nDatabase columns check:")
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'wastex.settings')
import django
django.setup()

from django.db import connection
cursor = connection.cursor()
cursor.execute("""
    SELECT column_name FROM information_schema.columns 
    WHERE table_name='images' 
    ORDER BY column_name
""")
columns = [row[0] for row in cursor.fetchall()]
expected = ['source_device', 'predicted_label', 'pi_upload_timestamp']
print(f"Columns: {len(columns)} total")
for col in expected:
    print(f"  {col}: {'✓' if col in columns else '✗'}")
