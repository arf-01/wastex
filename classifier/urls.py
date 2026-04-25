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
from .views.pi_api import api_pi_inference, api_pi_health
from .views.api_keys import profile as api_key_profile, regenerate_token
from .views.bin_api import api_bins
from .views import sync_api

urlpatterns = [
    # ── Edge Pages ──────────────────────────────────────────────────────
    path("edge/dashboard/", views.dashboard, name="edge_dashboard"),
    path("edge/upload/", views.upload, name="edge_upload"),
    path("edge/inspect/", views.inspect, name="edge_inspect"),

    # ── Master Pages ────────────────────────────────────────────────────
    path("master/dataset/", views.dataset_view, name="master_dataset"),
    path("master/training/", views.training_view, name="master_training"),

    # ── Classification ──────────────────────────────────────────────────
    path("classify/", views.classify, name="classify"),

    # ── Trash counter & Bins ────────────────────────────────────────────
    path("api/counts/", views.api_trash_counts, name="api_trash_counts"),
    path("api/history/", views.api_trash_history, name="api_trash_history"),
    path("api/bins/", api_bins, name="api_bins"),

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

    # ── Raspberry Pi Inference (Simple, reuses existing classify logic) ──
    path("api/pi/health/", api_pi_health, name="api_pi_health"),
    path("api/pi/inference/", api_pi_inference, name="api_pi_inference"),

    # ── API Key Management ──────────────────────────────────────────────
    path("edge/api-key/", api_key_profile, name="api_key_profile"),
    path("edge/api-key/regenerate/", regenerate_token, name="regenerate_token"),

    # ── Cloud Broker Sync APIs ──────────────────────────────────────────
    path("api/sync/ood/receive/", sync_api.api_receive_ood, name="sync_receive_ood"),
    path("api/sync/ood/pending/", sync_api.api_get_pending_images, name="sync_get_pending"),
    path("api/sync/model/release/", sync_api.api_cloud_release_model, name="sync_release_model"),
    path("api/sync/model/latest/", sync_api.api_get_latest_release, name="sync_get_latest_model"),
]

