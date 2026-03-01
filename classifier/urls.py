"""
URL configuration for the classifier app.

Route groups
------------
- Page views   : dashboard, upload, inspect, dataset browser.
- Classify API : POST endpoint for image classification.
- Trash API    : aggregated counts and time-series history.
- OOD API      : list / review / label out-of-distribution images.
- Dataset API  : versions, staging area, create/register versions, image browser.
"""

from django.shortcuts import redirect
from django.urls import path

from . import views

urlpatterns = [
    # Root redirect
    path("", lambda r: redirect("dashboard")),

    # ── Page views ──────────────────────────────────────────────────────
    path("dashboard/", views.dashboard, name="dashboard"),
    path("upload/", views.upload, name="upload"),
    path("inspect/", views.inspect, name="inspect"),
    path("dataset/", views.dataset_view, name="dataset"),
    path("training/", views.training_view, name="training"),

    # ── Classification ──────────────────────────────────────────────────
    path("classify/", views.classify, name="classify"),

    # ── Trash counter ───────────────────────────────────────────────────
    path("api/counts/", views.api_trash_counts, name="api_trash_counts"),
    path("api/history/", views.api_trash_history, name="api_trash_history"),

    # ── OOD images ──────────────────────────────────────────────────────
    path("api/ood/", views.api_ood_images, name="api_ood_images"),
    path("api/ood/<int:image_id>/review/", views.api_review_image, name="api_review_image"),
    path("api/ood/<int:image_id>/label/", views.api_label_image, name="api_label_image"),

    # ── Dataset versioning ──────────────────────────────────────────────
    path("api/classes/", views.api_classes, name="api_classes"),
    path("api/dataset/versions/", views.api_dataset_versions, name="api_dataset_versions"),
    path("api/dataset/active/", views.api_active_version, name="api_active_version"),
    path("api/dataset/set-active/", views.api_set_active_version, name="api_set_active_version"),
    path("api/dataset/staged/", views.api_staged_images, name="api_staged_images"),
    path("api/dataset/create-version/", views.api_create_version, name="api_create_version"),
    path("api/dataset/register-version/", views.api_register_version, name="api_register_version"),
    path("api/dataset/images/", views.api_dataset_images, name="api_dataset_images"),

    # ── Training ────────────────────────────────────────────────────────
    path("api/training/start/", views.api_training_start, name="api_training_start"),
    path("api/training/status/", views.api_training_status, name="api_training_status"),
    path("api/training/history/", views.api_training_history, name="api_training_history"),
    path("api/training/promote/", views.api_training_promote, name="api_training_promote"),
]
