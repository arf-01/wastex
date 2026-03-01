from django.shortcuts import render


def welcome(request):
    """Render the welcome page."""
    return render(request, 'welcome.html')
