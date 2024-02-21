import traceback
import pandas as pd
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

user_inputs = []
@api_view(['POST'])
def show_top3_image(request):
    global user_inputs
    # 여기부터
    print("넘어왔당")
    
    print(request.FILES)
    img_file = request.FILES.get('image')
    # print(img_file)
    # personal_color = request.FILES.get('personal_color')
    if img_file:
        try:
            image = Image.open(img_file)
            rgb_image = image.convert('RGB')
            output_io = BytesIO()
            rgb_image.save(output_io, format='JPEG')
            user_inputs.append(rgb_image)
            print("오류 처리??")
            csv_df = pd.read_csv('./laurant051/laurant051.csv', header=None, names=['index', 'position', 'img_url', 'shopping_url'])
            upper_idx=[]
            lower_idx=[]
            model = SimilarityModel(test_recommended, FeaturingModel())

            test_recommended = {
                "upper":[f"./laurant051/features/{i}.pt" for i in range(0, 1000+1)],
                "lower":[f"./laurant051/features/{i}.pt" for i in range(0, 1000+1)],
            }
            k=3
            topk_upper, similarity_result_upper = model(user_inputs, "upper", k)
            topk_lower, similarity_result_lower = model(user_inputs, "lower", k)
            new_topk_upper_path=[]
            new_topk_upper_shopping=[]
            new_topk_lower_path=[]
            new_topk_lower_shopping=[]
            for path in topk_upper:
                new_path=path.replace("features", "image").replace(".pt",".jpg")
                n = new_path.split('/')[-1].split('.')[0]
                matching_row = csv_df[csv_df['index'] == int(n)]                
                new_topk_upper_path.append(matching_row['img_url'].values[0])
                new_topk_upper_shopping.append(matching_row['shopping_url'].values[0])
            for path in topk_lower:
                new_path=path.replace("features", "image").replace(".pt",".jpg")
                n = new_path.split('/')[-1].split('.')[0]
                matching_row = csv_df[csv_df['index'] == int(n)]
                new_topk_lower_path.append(matching_row['img_url'].values[0])
                new_topk_lower_shopping.append(matching_row['shopping_url'].values[0])
                new_topk_lower_shopping.append(int(n))

            return JsonResponse({'count': k, 'topk_upper': new_topk_upper_path, 'topk_lower': new_topk_lower_path, 'topk_shopping_upper': new_topk_upper_shopping, 'topk_shopping_lower': new_topk_lower_shopping})
        except Exception as e:
            # 이미지 처리 중 예외가 발생한 경우 오류 응답 반환
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