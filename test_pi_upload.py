#!/usr/bin/env python3
"""
Test script for WasteX Pi inference endpoint with token authentication.

Usage:
    python test_pi_upload.py <image_path> <backend_url> <token> [bin_id]

Example:
    python test_pi_upload.py photo.jpg http://192.169.0.111:8000 a3f9b2c1d7e4f8a6b0c5d2e1f7a4b3c8 bin_01

Token is obtained by logging in to the WasteX dashboard and visiting:
    /classifier/edge/api-key/

The optional bin_id argument identifies this device; it will be registered
and its last_active timestamp updated on every health check and inference.
"""

import sys
import requests
from pathlib import Path
import json

def test_pi_inference(image_path: str, backend_url: str, token: str, bin_id: str = ""):
    """Test the Pi inference endpoint with token authentication."""
    
    image_file = Path(image_path)
    if not image_file.exists():
        print(f"❌ Error: Image file not found: {image_path}")
        sys.exit(1)
    
    # Prepare the request
    url = f"{backend_url.rstrip('/')}/api/pi/inference/"
    headers = {
        "Authorization": f"Token {token}",
    }
    
    print(f"\n📸 Sending image to Pi inference endpoint")
    print(f"   Backend: {backend_url}")
    print(f"   Endpoint: {url}")
    print(f"   Image: {image_file.name}")
    print(f"   Token: {token[:10]}...{token[-10:]}")
    if bin_id:
        print(f"   Bin ID: {bin_id}")
    
    try:
        with open(image_file, 'rb') as f:
            files = {'image': f}
            data = {'source': bin_id} if bin_id else {}
            response = requests.post(url, headers=headers, files=files, data=data, timeout=30)
        
        print(f"\n✓ Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            is_ood = result.get('ood', False)
            predicted_class = result.get('predicted_class')
            energy = result.get('energy')
            saved = result.get('saved_to_db', False)

            print(f"\n✓ Inference Results:")
            print(f"   Predicted Class : {predicted_class or 'N/A (OOD)'}")
            print(f"   Is OOD          : {is_ood}")
            print(f"   Energy Score    : {energy:.4f}" if energy is not None else "   Energy Score    : N/A")
            print(f"   Saved to DB     : {saved}")
            
            if is_ood:
                print(f"\n⚠️  Image was classified as Out-of-Distribution (OOD)")
                print(f"   It has been saved for operator review")
            else:
                print(f"\n✓ Image classified as: {predicted_class}")
                if bin_id:
                    print(f"   TrashItem event recorded for bin '{bin_id}'")
            
            return True
        
        elif response.status_code == 401:
            print(f"\n❌ Authentication failed (401)")
            print(f"   Token may be invalid or expired")
            print(f"   Response: {response.text}")
            return False
        
        elif response.status_code == 403:
            print(f"\n❌ Access forbidden (403)")
            print(f"   Your account may not have permission")
            print(f"   Response: {response.text}")
            return False
        
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Connection failed")
        print(f"   Could not reach {backend_url}")
        print(f"   Check that the backend is running and the URL is correct")
        return False
    
    except requests.exceptions.Timeout:
        print(f"\n❌ Timeout")
        print(f"   Request took too long (>30s)")
        return False
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False


def test_health_check(backend_url: str, token: str = "", bin_id: str = ""):
    """Test the authenticated health check endpoint.
    
    Sends the auth token so the backend can update the bin's last_active
    timestamp — identical behaviour to a real Pi heartbeat.
    """
    params = {}
    if bin_id:
        params['source'] = bin_id

    url = f"{backend_url.rstrip('/')}/api/pi/health/"
    headers = {"Authorization": f"Token {token}"} if token else {}
    
    print(f"\n🔍 Health Check (authenticated)")
    print(f"   URL: {url}")
    if bin_id:
        print(f"   Bin ID: {bin_id}")
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Backend is reachable")
            print(f"  Status      : {result.get('status')}")
            print(f"  Message     : {result.get('message')}")
            print(f"  Bin Tracked : {result.get('bin_tracked')}")
            return True
        elif response.status_code == 401:
            print(f"❌ Auth failed (401) — check your token")
            return False
        else:
            print(f"❌ Unexpected status: {response.status_code} — {response.text}")
            return False
    
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        return False


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python test_pi_upload.py <image_path> <backend_url> <token> [bin_id]")
        print("\nExample:")
        print("  python test_pi_upload.py photo.jpg http://192.169.0.111:8000 a3f9b2c1d7e4f8a6... bin_01")
        print("\n📋 To get your token:")
        print("  1. Log in to http://192.169.0.111:8000/")
        print("  2. Visit /edge/api-key/")
        print("  3. Copy your token")
        sys.exit(1)
    
    image_path = sys.argv[1]
    backend_url = sys.argv[2]
    token = sys.argv[3]
    bin_id = sys.argv[4] if len(sys.argv) > 4 else ""
    
    # Authenticated health check — registers/updates the bin's last_active
    if not test_health_check(backend_url, token=token, bin_id=bin_id):
        print("\n⚠️  Health check failed. Continuing anyway...")
    
    # Test inference — sends image + optional bin_id via 'source' field
    success = test_pi_inference(image_path, backend_url, token, bin_id=bin_id)
    
    sys.exit(0 if success else 1)
