import traceback
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
from cv_proj.pipeline.SimilarityModel import SimilarityModel
from cv_proj.pipeline.FeaturingModel import FeaturingModel

# Create your views here.

user_inputs = []
# @api_view(['GET', 'POST'])
@api_view(['POST'])
def process_image(request):
    print("넘어왔당")
    img_file = request.FILES.get('image')
    if img_file:
        try:
            image = Image.open(img_file)
            # print("요기부터")
            rgb_image = image.convert('RGB')
            output_io = BytesIO()
            rgb_image.save(output_io, format='JPEG')
            print("요기까지 오류 처리")
            # user_inputs = [rgb_image]
            test_recommended = {
                "upper":[f"../test/features/feature{i}.pt" for i in range(0, 14+1)],
                "lower":[],
            }
            model = SimilarityModel(test_recommended, FeaturingModel())
            # user_inputs = []
            for i in range(0, 5+1):
                user_inputs.append(Image.open(f"test/image/{i}.jpg").convert("RGB"))
            # print(user_inputs)
            topk, similarity_result = model(user_inputs, "upper")
            print("model 끝나고 돌아왔당")
            # print(topk)
            print("\n")
            # print(similarity_result)

            # type = "upper"
            # k = 3
            # personal_color = None
            # print("model : ")
            # result_images, similarity_scores = model(user_inputs, type, k, personal_color)
            # print("model : ")
            # print(result_images)
            # features = model(rgb_image, only_clothes=True)
            print("오류 처리??")
            
            return JsonResponse({
                'message': '이미지가 처리됨.',
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