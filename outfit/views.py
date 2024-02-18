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
# @api_view(['GET', 'POST'])
@api_view(['POST'])
def process_image(request):
    global user_inputs
    # print("넘어왔당")
    img_file = request.FILES.get('image')
    if img_file:
        try:
            image = Image.open(img_file)
            rgb_image = image.convert('RGB')
            output_io = BytesIO()
            rgb_image.save(output_io, format='JPEG')
            user_inputs.append(rgb_image)

            # print("요기까지 오류 처리")
            # user_inputs = [rgb_image]
            # test_recommended = {
            #     "upper":[f"../test/features/feature{i}.pt" for i in range(0, 14+1)],
            #     "lower":[],
            # }

            # 버튼 누르면 요고 실행
            # model = SimilarityModel(test_recommended, FeaturingModel())
            # topk, similarity_result = model(user_inputs, "upper")

            # user_inputs = []
            # for i in range(0, 5+1):
            #     user_inputs.append(Image.open(f"test/image/{i}.jpg").convert("RGB"))
            # print(user_inputs)
            # print("model 끝나고 돌아왔당")
            # print(topk)
            print("\n")
            # print(similarity_result)
            print("오류 처리??")
            
            return JsonResponse({
                'message': '이미지가 처리됨.',
                # 'user_inputs': user_inputs
                # 'result_images': result_images,
                # 'similarity_scores': similarity_scores
            })
        except Exception as e:
            # 이미지 처리 중 예외가 발생한 경우 오류 응답 반환
            # print("에러")
            print(traceback.format_exc())
            return JsonResponse({'error': '이미지 처리 중 오류 발생. 에러 메시지: ' + str(e)})
    else:
        # 이미지가 전송되지 않았을 경우 오류 응답 반환
        response_data = {'message': '이미지를 찾을 수 없음.'}
        return JsonResponse(response_data, status=status.HTTP_400_BAD_REQUEST)
    
@api_view(['GET'])
def show_top3_image(request):
    global user_inputs
    if user_inputs:
        try:
            # 버튼 누르면 요고 실행
            test_recommended = {
                "upper":[f"./laurant051/features/{i}.pt" for i in range(0, 14+1)],
                "lower":[f"./laurant051/features/{i}.pt" for i in range(0, 14+1)],
            }
            model = SimilarityModel(test_recommended, FeaturingModel())
            k=3
            topk_upper, similarity_result_upper = model(user_inputs, "upper", k)
            topk_lower, similarity_result_lower = model(user_inputs, "lower", k)
            new_topk_upper_path=[]
            new_topk_upper_shopping=[]
            new_topk_lower_path=[]
            new_topk_lower_shopping=[]
            csv_df = pd.read_csv('./laurant051/laurant051.csv', header=None, names=['index', 'position', 'img_url', 'shopping_url'])
            for path in topk_upper:
                new_path=path.replace("features", "image").replace(".pt",".jpg")
                n = new_path.split('/')[-1].split('.')[0]
                print(type(n))
                print(n)
                print(type(csv_df['index']))
                matching_row = csv_df[csv_df['index'] == int(n)]                
                print(matching_row)
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
            # 모델 처리 되지 않았을 경우 오류 응답 반환
            print(traceback.format_exc())
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