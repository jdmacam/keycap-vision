from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing, name='landing'),
    path('upload', views.upload, name='upload'),
    path('results', views.results, name='results'),
    path('upload_reload', views.upload_reload, name='upload_reload'),
    path('keeb_list', views.keeb_list, name='keeb_list'),
]