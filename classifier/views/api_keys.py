"""
API key management for edge devices (Raspberry Pi).

Users can generate their token on-demand and regenerate it via the profile page.
The token is required for all Pi API endpoints.
"""

from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from rest_framework.authtoken.models import Token


@login_required
def profile(request):
    """Show the current user's API token (if exists) and offer to generate/regenerate it.
    
    GET: Display profile with current token (or message to generate one)
    """
    user = request.user
    
    # Try to get existing token (may not exist yet)
    token = Token.objects.filter(user=user).first()
    
    context = {
        'token': token.key if token else None,
        'token_created': token.created if token else None,
        'has_token': token is not None,
        'active_page': 'api_key',
    }

    return render(request, 'classifier/api_key_profile.html', context)


@login_required
@require_http_methods(["POST"])
def regenerate_token(request):
    """Generate a new API token (on-demand) or regenerate existing one.
    
    If user has no token yet, creates one.
    If user has a token, deletes old one and creates new one.
    Returns the token as JSON.
    """
    user = request.user
    
    # Delete old token if exists
    Token.objects.filter(user=user).delete()
    
    # Create new token
    token = Token.objects.create(user=user)
    
    return JsonResponse({
        'success': True,
        'new_token': token.key,
        'message': 'New token generated successfully!' if token else 'Token generation failed.',
    })
