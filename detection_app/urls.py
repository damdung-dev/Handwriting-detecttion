from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict_file/', views.predict_file, name='predict_file'),
    path('predict_draw/', views.predict_draw, name='predict_draw'),
]