from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('classify/', views.classify, name='classify'),
    path('api/docs/', views.api_docs, name='api_docs'),
    path('dashboard/', views.dashboard, name='dashboard'),
]
