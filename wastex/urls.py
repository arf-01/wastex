"""
Root URL configuration for the WasteX project.

All classifier functionality lives under ``/classifier/``.
The Django admin is available at ``/admin/``.
"""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.shortcuts import redirect
from django.urls import include, path
from classifier.views.pi_api import api_pi_inference, api_pi_health
import classifier.views.pages

urlpatterns = [
    path("", classifier.views.pages.root_redirect, name="root_redirect"),
    path("accounts/", include("django.contrib.auth.urls")),
    path("", include("classifier.urls")),
    path("admin/", admin.site.urls),
]

# Serve uploaded media and dataset images in development
# Always enabled for local roles (EDGE/MASTER); cloud uses S3
if settings.SITE_ROLE != 'CLOUD':
    from django.views.static import serve
    from django.urls import re_path
    urlpatterns += [
        re_path(r'^media/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT}),
        re_path(r'^datasets/(?P<path>.*)$', serve, {'document_root': settings.DATASETS_ROOT}),
    ]
elif settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static("/datasets/", document_root=settings.DATASETS_ROOT)

