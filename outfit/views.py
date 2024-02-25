import traceback
import pandas as pd
import json
import base64
import os
from django.conf import settings
from django.shortcuts import render
from rest_framework import viewsets, permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.generics import get_object_or_404
from .models import OutfitInfo
from .serializers import OutfitSerializer

from PIL import Image
from io import BytesIO
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.http import HttpResponse
from django.http import JsonResponse
from .pipeline.SimilarityModel import SimilarityModel
from .pipeline.FeaturingModel import FeaturingModel

# Create your views here.
male_upper_idx=[]
male_lower_idx=[]
female_upper_idx=[]
female_lower_idx=[]
male_recommended={}
female_recommended={}
model_initialized = False
user_inputs = []
model=''
csv_df=''
DIR = settings.BASE_DIR

def initialize_model(gender):
    global model_initialized, model, female_upper_idx, female_lower_idx, male_upper_idx, male_lower_idx, male_recommended, female_recommended, female_df, male_df
    # D:\coordi_recommend\outfit_recommendation
    # print(settings.BASE_DIR) 
    if not model_initialized:
        female_df = pd.read_csv(os.path.join(DIR,'slowand/slowand.csv'), header=None, names=['index', 'position', 'img_url', 'shopping_url', "title"], index_col='index', encoding='UTF8')
        male_df = pd.read_csv(os.path.join(DIR,'laurant051/laurant051.csv'), header=None, names=['index', 'position', 'img_url', 'shopping_url', "title"], index_col='index', encoding='UTF8')
        for idx, row in female_df.iterrows():
            if row['position']==0:
                female_upper_idx.append(idx)
            else:
                female_lower_idx.append(idx)
        print(female_upper_idx)
        for idx, row in male_df.iterrows():
            if row['position']==0:
                male_upper_idx.append(idx)
            else:
                male_lower_idx.append(idx)
        male_recommended = {
            "upper":[os.path.join(DIR, f"laurant051/features/{i}.pt") for i in male_upper_idx],
            "lower":[os.path.join(DIR, f"laurant051/features/{i}.pt") for i in male_lower_idx],
        }
        female_recommended = {
            "upper": [os.path.join(DIR, f"slowand/features/{i}.pt") for i in female_upper_idx],
            "lower": [os.path.join(DIR, f"slowand/features/{i}.pt") for i in female_lower_idx],
        }
        model = SimilarityModel(male_recommended, female_recommended, FeaturingModel())
        if gender == 'female':
            csv_df = female_df
        else:
            csv_df = male_df
        model_initialized = True
        return model, csv_df
    else:
        if gender == 'female':
            csv_df = female_df
        else:
            csv_df = male_df
        model_initialized = True
        print("Model has already been initialized.")
        return model, csv_df

@api_view(['POST'])
def show_top3_image(request):
    global model, user_inputs, female_upper_idx, female_lower_idx, male_upper_idx, male_lower_idx, female_recommended, male_recommended
    request_data = request.data
    gender = request_data.get('gender')
    color = request_data.get('color')
    base64_images = request_data.get('images', [])
    print("넘어왔당")
    if base64_images:
        try:
            for base64_image in base64_images:
                if base64_image.startswith('data:image/jpeg;base64,'):
                    base64_image = base64_image.split('base64,')[-1]
                elif base64_image.startswith('data:image/png;base64,'):
                    base64_image = base64_image.split('base64,')[-1]

                image_data = base64.b64decode(base64_image)
                image = Image.open(BytesIO(image_data))
                user_inputs.append(image)
            model, csv_df = initialize_model(gender)
            k=3
            topk_upper, similarity_result_upper = model(user_inputs, "upper", gender, k, color)
            topk_lower, similarity_result_lower = model(user_inputs, "lower", gender, k, color)
            print(topk_upper)
            new_topk_upper_path=[]
            new_topk_upper_shopping=[]
            new_topk_lower_path=[]
            new_topk_lower_shopping=[]
            for path in topk_upper:
                # print(path)
                new_path=path.replace("features", "image").replace(".pt",".jpg")
                n = new_path.split('/')[-1].split('.')[0]
                matching_row = csv_df.loc[[int(n)]] #csv_df[csv_df['index'] == int(n)]
                # print("upper")
                # print(n)
                new_topk_upper_path.append(matching_row['img_url'].values[0])
                new_topk_upper_shopping.append(matching_row['shopping_url'].values[0])
                print(new_topk_upper_path)
            for path in topk_lower:
                new_path=path.replace("features", "image").replace(".pt",".jpg")
                n = new_path.split('/')[-1].split('.')[0]
                matching_row = csv_df.loc[[int(n)]] #csv_df[csv_df['index'] == int(n)]
                # print("lower")
                # print(n)
                new_topk_lower_path.append(matching_row['img_url'].values[0])
                new_topk_lower_shopping.append(matching_row['shopping_url'].values[0])
                # new_topk_lower_shopping.append(int(n))

            return JsonResponse({'count': k, 'topk_upper': new_topk_upper_path, 'topk_lower': new_topk_lower_path, 'topk_shopping_upper': new_topk_upper_shopping, 'topk_shopping_lower': new_topk_lower_shopping})
        except Exception as e:
            # 이미지 처리 중 예외가 발생한 경우 오류 응답 반환
            traceback.print_exc()
            return JsonResponse({'error': '모델 처리 중 오류 발생. 에러 메시지: ' + str(e)})
    else:
        # 이미지가 전송되지 않았을 경우 오류 응답 반환
        response_data = {'message': '이미지 파일 없음.'}
        return JsonResponse(response_data, status=status.HTTP_400_BAD_REQUEST)
        

@api_view(['GET'])
class OutfitsAPI(APIView):
    def get(self, request):
        request.body
        outfits = OutfitInfo.objects.all()
        serializer = OutfitSerializer(outfits, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    def post(self, request):
        # 데이터 처리하고 image_url, shopping_url 가져오기
        count = 3
        response_data = {
            'count': count,
            'upper': [{'image_url': 'abc', 'shopping_url': 'Abc'}] * count,
            'lower': [{'image_url': 'abc', 'shopping_url': 'Abc'}] * count,
        }
        return Response(response_data, status=status.HTTP_201_CREATED)
    

class OutfitAPI(APIView):    
    def get(self, request, oid):
        outfit = get_object_or_404(OutfitInfo, oid=oid)
        serializer = OutfitSerializer(outfit)
        return Response(serializer.data, status=status.HTTP_200_OK)
