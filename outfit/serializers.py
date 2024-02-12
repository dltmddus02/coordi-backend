from rest_framework import serializers
from .models import OutfitInfo

class OutfitSerializer(serializers.ModelSerializer):
    class Meta:
        model = OutfitInfo
        fields = ['oid', 'count', 'image_data', 'p_color']