from django.db import models

# Create your models here.
class OutfitInfo(models.Model):
    oid = models.IntegerField(primary_key=True) 
    count = models.IntegerField() # 올린 사진 개수
    image_data = models.ImageField() # 이미지 데이터
    # image_data = models.TextField(null=True) # 이미지 데이터
    p_color = models.IntegerField() # 퍼컬 번호