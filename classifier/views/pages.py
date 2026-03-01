"""
HTML page views — render templates for the main UI pages.
"""

from __future__ import annotations

import json

from django.shortcuts import render
from django.views.decorators.http import require_GET

from .helpers import get_all_class_names


@require_GET
def dashboard(request):
    """Render the main dashboard page."""
    class_names = get_all_class_names()
    return render(request, "classifier/dashboard.html", {
        "class_names": class_names,
        "class_names_json": json.dumps(class_names),
    })


@require_GET
def upload(request):
    """Render the image upload / classification page."""
    class_names = get_all_class_names()
    return render(request, "classifier/upload.html", {
        "class_names": class_names,
        "class_names_json": json.dumps(class_names),
    })


@require_GET
def inspect(request):
    """Render the OOD image inspection / labelling page."""
    class_names = get_all_class_names()
    return render(request, "classifier/inspect.html", {
        "class_names": class_names,
        "class_names_json": json.dumps(class_names),
    })


@require_GET
def dataset_view(request):
    """Render the dataset version browser page."""
    class_names = get_all_class_names()
    return render(request, "classifier/dataset.html", {
        "class_names": class_names,
        "class_names_json": json.dumps(class_names),
    })


@require_GET
def training_view(request):
    """Render the training management page."""
    class_names = get_all_class_names()
    return render(request, "classifier/training.html", {
        "class_names": class_names,
        "class_names_json": json.dumps(class_names),
    })
