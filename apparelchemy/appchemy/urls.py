from django.urls import path
from . import views
from django.conf.urls import url

urlpatterns = [
    path('', views.home, name='appchemy-home'),
    url('check/$', views.check, name='appchemy-check'),
    url('recommend/$', views.recommend, name='appchemy-recommend'),
]