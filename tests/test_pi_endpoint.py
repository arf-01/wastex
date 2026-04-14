#!/usr/bin/env python
"""Test the Pi inference endpoint"""

import requests
import json
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000/classifier/api/pi/inference/"

# Test 1: POST without image (should return 400)
print("Test 1: POST without image file")
try:
    response = requests.post(BASE_URL)
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.text}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n" + "="*60 + "\n")

# Test 2: POST with test image
print("Test 2: POST with test image")
test_image_path = Path("datasets/v1/class_0/000001.jpg")
if test_image_path.exists():
    try:
        with open(test_image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(BASE_URL, files=files)
            print(f"  Status: {response.status_code}")
            print(f"  Response: {response.json()}")
    except Exception as e:
        print(f"  ERROR: {e}")
else:
    print(f"  Test image not found: {test_image_path}")

print("\nTest complete!")
