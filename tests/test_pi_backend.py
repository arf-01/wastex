#!/usr/bin/env python3
"""Test Pi connectivity to backend"""
import requests
import sys
import time

time.sleep(2)  # Give server time to start

try:
    print("🔌 Testing connection to backend...")
    r = requests.get('http://192.169.0.111:8000/classifier/api/pi/health/', timeout=5)
    print(f"✅ Status Code: {r.status_code}")
    print(f"✅ Response: {r.json()}")
    sys.exit(0)
except requests.exceptions.ConnectionError as e:
    print(f"❌ Connection Error: Cannot reach backend")
    print(f"   Details: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {type(e).__name__}")
    print(f"   Details: {e}")
    sys.exit(1)
