from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('upload/', views.upload_file, name='upload_file'),
    path('files/', views.file_selection, name='file_selection'),
    path('results/<int:file_id>/', views.results_page, name='results_page'),
    path('summary/overall/<int:file_id>/', views.overall_summary_page, name='overall_summary_page'),
    path('summary/questionwise/<int:file_id>/', views.question_summary_page, name='question_summary_page'),
]
