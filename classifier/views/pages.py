"""
HTML page views — render templates for the main UI pages.
"""

from __future__ import annotations

import json

from django.shortcuts import render
from django.views.decorators.http import require_GET

from .helpers import get_all_class_names


def _page_context(active_page: str) -> dict:
    """Build the common template context shared by every page."""
    class_names = get_all_class_names()
    return {
        "class_names": class_names,
        "class_names_json": json.dumps(class_names),
        "active_page": active_page,
    }


@require_GET
def dashboard(request):
    """Render the main dashboard page."""
    return render(request, "classifier/dashboard.html", _page_context("dashboard"))


@require_GET
def upload(request):
    """Render the image upload / classification page."""
    return render(request, "classifier/upload.html", _page_context("upload"))


@require_GET
def inspect(request):
    """Render the OOD image inspection / labelling page."""
    return render(request, "classifier/inspect.html", _page_context("inspect"))


@require_GET
def dataset_view(request):
    """Render the dataset version browser page."""
    return render(request, "classifier/dataset.html", _page_context("dataset"))


@require_GET
def training_view(request):
    """Render the training management page."""
    return render(request, "classifier/training.html", _page_context("training"))
