#!/usr/bin/env python3
"""
Test script to verify the Pi Real-time Inference API works correctly.

Run this on your local machine to test the backend endpoints.

Usage:
    python test_pi_api.py <backend_url> <test_image_path>

Example:
    python test_pi_api.py http://localhost:8000 sample_image.jpg
"""

import sys
import requests
import json
from pathlib import Path


def test_health_check(backend_url):
    """Test the health check endpoint."""
    print("\n" + "="*70)
    print("1️⃣  Testing Health Check")
    print("="*70)
    
    try:
        response = requests.get(f"{backend_url}/classifier/api/pi/health/", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health Check PASSED")
            print(f"   Status: {data.get('status')}")
            print(f"   Timestamp: {data.get('timestamp')}")
            return True
        else:
            print(f"❌ Health Check FAILED")
            print(f"   Status Code: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    
    except Exception as e:
        print(f"❌ Health Check FAILED: {e}")
        return False


def test_single_inference(backend_url, image_path):
    """Test single image inference."""
    print("\n" + "="*70)
    print("2️⃣  Testing Single Image Inference")
    print("="*70)
    
    if not Path(image_path).exists():
        print(f"❌ Image file not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {'source': 'test_pi'}
            
            response = requests.post(
                f"{backend_url}/classifier/api/pi/inference/",
                files=files,
                data=data,
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Single Inference PASSED")
            print(f"   Status: {result.get('status')}")
            print(f"   Predictions: {json.dumps(result.get('predictions'), indent=2)}")
            print(f"   Image ID: {result.get('image_id')}")
            print(f"   Timestamp: {result.get('timestamp')}")
            return True
        else:
            print(f"❌ Single Inference FAILED")
            print(f"   Status Code: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    
    except Exception as e:
        print(f"❌ Single Inference FAILED: {e}")
        return False


def test_batch_inference(backend_url, image_path):
    """Test batch image inference."""
    print("\n" + "="*70)
    print("3️⃣  Testing Batch Image Inference")
    print("="*70)
    
    if not Path(image_path).exists():
        print(f"❌ Image file not found: {image_path}")
        return False
    
    try:
        # Create a tuple of multiple image files (same image twice for testing)
        with open(image_path, 'rb') as f1:
            with open(image_path, 'rb') as f2:
                files = [
                    ('images', ('image1.jpg', f1, 'image/jpeg')),
                    ('images', ('image2.jpg', f2, 'image/jpeg')),
                ]
                data = {'source': 'test_pi'}
                
                response = requests.post(
                    f"{backend_url}/classifier/api/pi/batch-inference/",
                    files=files,
                    data=data,
                    timeout=30
                )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch Inference PASSED")
            print(f"   Status: {result.get('status')}")
            print(f"   Total Processed: {result.get('total_processed')}")
            print(f"   Results: {len(result.get('results', []))} images processed")
            return True
        else:
            print(f"❌ Batch Inference FAILED")
            print(f"   Status Code: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    
    except Exception as e:
        print(f"❌ Batch Inference FAILED: {e}")
        return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "Pi Real-time Inference API Test" + " "*21 + "║")
    print("╚" + "="*68 + "╝")
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("\nUsage: python test_pi_api.py <backend_url> [image_path]")
        print("\nExample:")
        print("  python test_pi_api.py http://localhost:8000 sample_image.jpg")
        sys.exit(1)
    
    backend_url = sys.argv[1].rstrip('/')
    image_path = sys.argv[2] if len(sys.argv) > 2 else 'test_image.jpg'
    
    print(f"\n📍 Backend URL: {backend_url}")
    print(f"📷 Test Image: {image_path}")
    
    results = []
    
    # Test 1: Health Check
    results.append(test_health_check(backend_url))
    
    # Test 2: Single Inference (requires image)
    if Path(image_path).exists():
        results.append(test_single_inference(backend_url, image_path))
    else:
        print(f"\n⚠️  Skipping single image test (image not found: {image_path})")
    
    # Test 3: Batch Inference (requires image)
    if Path(image_path).exists():
        results.append(test_batch_inference(backend_url, image_path))
    else:
        print(f"\n⚠️  Skipping batch image test (image not found: {image_path})")
    
    # Summary
    print("\n" + "="*70)
    print("📊 Test Summary")
    print("="*70)
    passed = sum(results)
    total = len(results)
    print(f"✅ Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Backend is ready for Pi uploads.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above.")
    
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
