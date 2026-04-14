from django.urls import path
from . import views

urlpatterns = [
    # Landing Page
    path('', views.home, name='home'),
    
    # Module 1
    path('module1/', views.module1_workspace, name='module1'),
    
    # Module 2 (Visual Analytics)
    path('module2/', views.module2_visual_analytics, name='module2'),
    # Module 3 (Auto ML)
    path('module3/', views.module3_automl, name='module3'),
]