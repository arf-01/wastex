from functools import wraps
from django.contrib.auth.decorators import user_passes_test
from django.core.exceptions import PermissionDenied

def edge_required(view_func):
    """
    Decorator for views that checks that the user is an Edge user or Superuser.
    """
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if request.user.is_superuser or request.user.groups.filter(name='EdgeUsers').exists():
            return view_func(request, *args, **kwargs)
        raise PermissionDenied("You must be an Edge User to view this page.")
    return _wrapped_view

def master_required(view_func):
    """
    Decorator for views that checks that the user is a Master user or Superuser.
    """
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if request.user.is_superuser or request.user.groups.filter(name='MasterUsers').exists():
            return view_func(request, *args, **kwargs)
        raise PermissionDenied("You must be a Master User to access the Retraining Pipeline.")
    return _wrapped_view
