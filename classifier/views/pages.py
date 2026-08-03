"""
HTML page views — render templates for the main UI pages.
"""

from __future__ import annotations

import json

from django.shortcuts import render, redirect
from django.views.decorators.http import require_GET

from .helpers import get_all_class_names
from django.contrib.auth.decorators import login_required
from classifier.decorators import edge_required, master_required


from django.conf import settings

def _page_context(active_page: str) -> dict:
    """Build the common template context shared by every page."""
    class_names = get_all_class_names()
    return {
        "class_names": class_names,
        "class_names_json": json.dumps(class_names),
        "active_page": active_page,
        "SITE_ROLE": settings.SITE_ROLE,  # Pass role to UI
    }

@login_required
def root_redirect(request):
    """Redirect users to the appropriate landing page based on group membership or SITE_ROLE."""
    if request.user.is_superuser:
        if settings.SITE_ROLE == 'MASTER':
            return redirect('master_dataset')
        return redirect('edge_dashboard')
        
    if request.user.groups.filter(name='MasterUsers').exists():
        return redirect('master_dataset')
    elif request.user.groups.filter(name='EdgeUsers').exists():
        return redirect('edge_dashboard')
        
    # Fallback to site role
    if settings.SITE_ROLE == 'MASTER':
        return redirect('master_dataset')
    return redirect('edge_dashboard')

@login_required
@edge_required
@require_GET
def dashboard(request):
    """Render the main dashboard page."""
    return render(request, "classifier/dashboard.html", _page_context("dashboard"))


@login_required
@edge_required
@require_GET
def upload(request):
    """Render the image upload / classification page."""
    return render(request, "classifier/upload.html", _page_context("upload"))


@login_required
@edge_required
@require_GET
def inspect(request):
    """Render the OOD image inspection / labelling page."""
    return render(request, "classifier/inspect.html", _page_context("inspect"))


@login_required
@master_required
@require_GET
def dataset_view(request):
    """Render the dataset version browser page."""
    return render(request, "classifier/dataset.html", _page_context("dataset"))


@login_required
@master_required
@require_GET
def training_view(request):
    """Render the training management page."""
    return render(request, "classifier/training.html", _page_context("training"))
