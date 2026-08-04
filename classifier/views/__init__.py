"""
View package for the WasteX classifier app.

Modules
-------
helpers.py        – Shared constants, utilities, and helper functions.
pages.py          – HTML page views (dashboard, upload, inspect, dataset).
classification.py – Image upload and model inference endpoint.
trash_api.py      – Trash counter read APIs (counts, history).
ood_api.py        – OOD image listing, review, and labelling APIs.
dataset_api.py    – Dataset versioning APIs (register, create, browse).
training_api.py   – Training run lifecycle APIs (start, status, promote).
"""

# Re-export all views so urls.py can do: from .views import dashboard, classify, …
from .pages import dashboard, upload, inspect, dataset_view  # noqa: F401
from .classification import classify                                  # noqa: F401
from .trash_api import api_trash_counts, api_trash_history            # noqa: F401
from .ood_api import api_ood_images, api_review_image, api_label_image  # noqa: F401

