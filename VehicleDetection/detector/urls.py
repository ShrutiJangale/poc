from django.urls import path
from . import views

app_name = 'detector'

urlpatterns = [
    path('', views.home, name='home'),
    path('analysis_stream/', views.video_analysis_stream, name='video_analysis_stream'),
    path('get_counts/', views.get_analysis_counts, name='get_analysis_counts'),
]
