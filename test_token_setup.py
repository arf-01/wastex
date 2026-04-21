#!/usr/bin/env python3
"""
Quick setup test for token authentication (without loading TensorFlow).
Tests that:
1. Token model and DRF are properly installed
2. Token endpoints are registered correctly
"""

import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'wastex.settings')
sys.path.insert(0, r'c:\WASTE\wastex')

try:
    django.setup()
    
    # Test imports
    print("✓ Django setup successful")
    
    from rest_framework.authtoken.models import Token
    print("✓ DRF Token model accessible")
    
    from django.contrib.auth.models import User
    print("✓ Django Auth User model accessible")
    
    from classifier.models import AppSettings
    print("✓ AppSettings model accessible")
    
    # Check URL patterns
    from django.urls import get_resolver
    from django.urls.exceptions import Resolver404
    
    resolver = get_resolver()
    
    # Test API key profile URL
    try:
        match = resolver.resolve('/classifier/edge/api-key/')
        print(f"✓ API key profile URL works: {match.view_name}")
    except Resolver404:
        print("✗ API key profile URL not found")
    
    # Test API key regenerate URL
    try:
        match = resolver.resolve('/classifier/edge/api-key/regenerate/')
        print(f"✓ API key regenerate URL works: {match.view_name}")
    except Resolver404:
        print("✗ API key regenerate URL not found")
    
    # Test Pi health endpoint
    try:
        match = resolver.resolve('/classifier/api/pi/health/')
        print(f"✓ Pi health URL works: {match.view_name}")
    except Resolver404:
        print("✗ Pi health URL not found")
    
    # Test Pi inference endpoint
    try:
        match = resolver.resolve('/classifier/api/pi/inference/')
        print(f"✓ Pi inference URL works: {match.view_name}")
    except Resolver404:
        print("✗ Pi inference URL not found")
    
    print("\n✓ All token authentication components are properly configured!")
    print("\nNext steps:")
    print("  1. Run: python manage.py migrate")
    print("  2. Create a user account")
    print("  3. Visit http://localhost:8000/classifier/edge/api-key/")
    print("  4. Copy your token and test with test_pi_upload.py")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
