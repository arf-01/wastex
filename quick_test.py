import requests
import time
import sys

# Give server time to start
time.sleep(3)

print("Testing Django endpoints...")
print("-" * 50)

tests = [
    ('GET', 'http://127.0.0.1:8000/', 'Root redirect'),
    ('GET', 'http://127.0.0.1:8000/classifier/dashboard/', 'Dashboard'),
    ('GET', 'http://127.0.0.1:8000/classifier/api/counts/', 'API: Counts'),
    ('GET', 'http://127.0.0.1:8000/classifier/api/ood/', 'API: OOD images'),
    ('GET', 'http://127.0.0.1:8000/classifier/api/pi/health/', 'API: Pi Health'),
]

passed = 0
failed = 0

for method, url, desc in tests:
    try:
        response = requests.request(method, url, timeout=5)
        if response.status_code == 200:
            print(f"✓ {desc:30} - {response.status_code}")
            passed += 1
        else:
            print(f"✗ {desc:30} - {response.status_code}")
            failed += 1
    except Exception as e:
        print(f"✗ {desc:30} - {str(e)[:40]}")
        failed += 1

print("-" * 50)
print(f"Results: {passed} passed, {failed} failed")

if failed > 0:
    sys.exit(1)
