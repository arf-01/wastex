import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'wastex.settings')
import django
django.setup()

from django.db import connection
cursor = connection.cursor()

# Get all columns in the images table
cursor.execute("""
    SELECT column_name FROM information_schema.columns 
    WHERE table_name='images' 
    ORDER BY column_name
""")

columns = [row[0] for row in cursor.fetchall()]
print("Columns in 'images' table:")
for col in columns:
    print(f"  - {col}")

# Check for specific columns we care about
expected = ['source_device', 'predicted_label', 'pi_upload_timestamp']
print("\nTarget columns status:")
for col in expected:
    exists = col in columns
    print(f"  {col}: {'✓ EXISTS' if exists else '✗ MISSING'}")
