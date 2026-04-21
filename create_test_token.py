#!/usr/bin/env python
"""
Test script to create a user and verify token authentication.
"""
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'wastex.settings')
django.setup()

from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token

# Create a test user if they don't exist
username = 'testuser'
email = 'test@example.com'
password = 'testpass123'

user, created = User.objects.get_or_create(
    username=username,
    defaults={'email': email}
)

if created:
    user.set_password(password)
    user.save()
    print(f"✓ Created user: {username}")
else:
    print(f"✓ User already exists: {username}")

# Get or create token
token, created = Token.objects.get_or_create(user=user)

print(f"\n📋 Token Information:")
print(f"   Username: {username}")
print(f"   Password: {password}")
print(f"   Token: {token.key}")
print(f"   Created: {token.created}")

print(f"\n🔗 Test URLs:")
print(f"   Login: http://localhost:8000/admin/login/")
print(f"   API Key Profile: http://localhost:8000/classifier/edge/api-key/")
print(f"   Health Check: http://localhost:8000/classifier/api/pi/health/")

print(f"\n💾 Use this token for Pi requests:")
print(f"   curl -X POST http://localhost:8000/classifier/api/pi/inference/ \\")
print(f"        -H 'Authorization: Token {token.key}' \\")
print(f"        -F 'image=@photo.jpg'")
