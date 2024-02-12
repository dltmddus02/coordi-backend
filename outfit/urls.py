from django.urls import path, include
from . import views
from .views import OutfitAPI, OutfitsAPI, process_image

urlpatterns = [
    # path("outfits/", OutfitsAPI.as_view()),
    # path("outfit/class/<int:oid>/", OutfitAPI.as_view()),
    # path("outfit/", views.process_image, name='process_image'),
    path("api/upload/", views.process_image, name='process_image'),
]