import torch
import torchvision
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from torchvision.transforms.functional import to_pil_image, crop
import torch.nn as nn
from typing import *
from enum import Enum
from colorsys import rgb_to_hsv, hsv_to_rgb
from pipeline.FeaturingModel import FeaturingModel, ClothClassificationModel

LAST_ACTIVATION_VOLUME = "last_activation_volume"
GRAM_MATRIX = "gram_matrix"
AVERAGE_RGB = "average_rgb"

LOWER_DEFAULT_COLOR = [(0, 0, 0), (255, 255, 255), (156, 156, 155), (217, 217, 215), (83, 86, 91), (254, 255, 239), (0, 31, 98), (61, 63, 107), (97, 134, 176), (38, 58, 84), (35, 40, 51), (33, 35, 34)]
PERSONAL_COLOR_RGB = {
    "WSB":[(215, 86, 116), (240, 95, 86), (252, 141, 60), (249, 187, 43), (217, 199, 27), (119, 187, 76), (3, 165, 131), (1, 148, 160), (1, 125, 175), (77, 115, 186), (139, 99, 172), (171, 87, 146), (181, 24, 78), (221, 55, 55), (230, 109, 0), (238, 172, 1), (201, 187, 1), (74, 163, 21), (2, 140, 105), (1, 124, 140), (0, 87, 146), (1, 79, 157), (102, 61, 140), (137, 44, 114)],
    "WSL":[(234, 185, 186), (233, 186, 170), (240, 205, 170), (240, 225, 182), (216, 214, 168), (164, 207, 183), (158, 205, 201), (165, 202, 216), (168, 183, 206), (183, 179, 204), (198, 175, 196), (223, 188, 199), (243, 140, 143), (255, 158, 125), (251, 184, 105), (237, 211, 103), (203, 202, 96), (115, 200, 156), (63, 171, 165), (82, 166, 192), (100, 145, 192), (142, 134, 189), (172, 126, 172), (217, 129, 149)],
    "WAD":[(159, 26, 50), (166, 58, 26), (170, 102, 2), (164, 143, 1), (112, 125, 1), (0, 111, 61), (1, 97, 92), (2, 81, 110), (1, 63, 116), (51, 51, 114), (88, 37, 95), (128, 30, 75), (100, 41, 45), (110, 54, 37), (110, 74, 24), (106, 94, 26), (83, 85, 22), (24, 80, 55), (0, 68, 65), (1, 66, 81), (19, 51, 76), (48, 45, 76), (65, 41, 67), (92, 43, 62), (60, 47, 46), (66, 60, 47), (65, 59, 46), (38, 52, 52), (44, 48, 67), (38, 40, 48), (45, 41, 44)],
    "WAM":[(185, 163, 166), (189, 162, 157), (201, 179, 161), (192, 183, 155), (185, 184, 154), (146, 175, 166), (148, 174, 176), (149, 173, 181), (147, 156, 171), (156, 154, 168), (163, 151, 163), (184, 163, 170), (200, 120, 123), (213, 137, 111), (213, 159, 97), (204, 182, 99), (174, 172, 95), (102, 169, 135), (73, 155, 149), (73, 138, 158), (89, 119, 155), (118, 111, 153), (139, 105, 139), (174, 111, 132), (109, 88, 89), (110, 88, 84), (119, 102, 86), (112, 104, 82), (110, 105, 81), (86, 110, 103), (82, 112, 114), (76, 96, 102), (77, 81, 93), (85, 80, 90), (89, 80, 89), (107, 89, 95), (156, 84, 87), (169, 97, 75), (169, 120, 61), (161, 142, 65), (133, 133, 59), (64, 130, 96), (27, 117, 112), (29, 100, 119), (51, 84, 115), (81, 77, 114), (102, 70, 103), (133, 76, 94)],
    "CSL":[(234, 185, 186), (233, 186, 170), (240, 205, 170), (240, 225, 182), (216, 214, 168), (164, 207, 183), (158, 205, 201), (165, 202, 216), (168, 183, 206), (183, 179, 204), (198, 175, 196), (223, 188, 199), (243, 140, 143), (255, 158, 125), (251, 184, 105), (237, 211, 103), (203, 202, 96), (115, 200, 156), (63, 171, 165), (82, 166, 192), (100, 145, 192), (142, 134, 189), (172, 126, 172), (217, 129, 149)],
    "CSM":[(185, 163, 166), (189, 162, 157), (201, 179, 161), (192, 183, 155), (185, 184, 154), (146, 175, 166), (148, 174, 176), (149, 173, 181), (147, 156, 171), (156, 154, 168), (163, 151, 163), (184, 163, 170), (200, 120, 123), (213, 137, 111), (213, 159, 97), (204, 182, 99), (174, 172, 95), (102, 169, 135), (73, 155, 149), (73, 138, 158), (89, 119, 155), (118, 111, 153), (139, 105, 139), (174, 111, 132), (109, 88, 89), (110, 88, 84), (119, 102, 86), (112, 104, 82), (110, 105, 81), (86, 110, 103), (82, 112, 114), (76, 96, 102), (77, 81, 93), (85, 80, 90), (89, 80, 89), (107, 89, 95), (156, 84, 87), (169, 97, 75), (169, 120, 61), (161, 142, 65), (133, 133, 59), (64, 130, 96), (27, 117, 112), (29, 100, 119), (51, 84, 115), (81, 77, 114), (102, 70, 103), (133, 76, 94)],
    "CWB":[(194, 52, 69), (199, 81, 45), (220, 139, 7), (209, 183, 0), (153, 164, 1), (1, 137, 81), (0, 123, 117), (0, 106, 138), (0, 87, 146), (75, 72, 140), (116, 58, 122), (160, 54, 100), (181, 24, 78), (221, 55, 55), (230, 109, 0), (238, 172, 1), (201, 187, 1), (74, 163, 21), (2, 140, 105), (1, 124, 140), (0, 87, 146), (1, 79, 157), (102, 61, 140), (137, 44, 114)],
    "CWD":[(159, 26, 50), (166, 58, 26), (170, 102, 2), (164, 143, 1), (112, 125, 1), (0, 111, 61), (1, 97, 92), (2, 81, 110), (1, 63, 116), (51, 51, 114), (88, 37, 95), (128, 30, 75), (100, 41, 45), (110, 54, 37), (110, 74, 24), (106, 94, 26), (83, 85, 22), (24, 80, 55), (0, 68, 65), (1, 66, 81), (19, 51, 76), (48, 45, 76), (65, 41, 67), (92, 43, 62), (60, 47, 46), (66, 60, 47), (65, 59, 46), (38, 52, 52), (44, 48, 67), (38, 40, 48), (45, 41, 44)]
}

class SimilarityModel:
    def __init__(self,
                 recommended: Dict[str, List[str]],
                 featuring_model: FeaturingModel,
                 useGPU: bool = False,
                 alpha: Tuple[int, int, int] = (1, 1, 1)):
        """
        유사도를 계산하여 추천 랭킹을 받는 클래스의 생성자입니다.


        :param recommended: 상의, 하의에 대한 정보와 각 옷의 특징 파일 경로
                            예를 들어, recommended = {"upper":["./pipeline/features/feature0.pt", "./pipeline/features/feature1.pt"], "lower":["./pipeline/features/feature2.pt"]}
        :param featuring_model: 유저의 입력 사진에 대해 특징 추출하는 모델.
        :param useGPU: 연산 시에 GPU를 이용할 것인가.
        :param alpha: 각 유사도의 가중합 계산 시 이용되는 가중치 상수입니다.
        """
        self.cpu_device = torch.device("cpu")
        self.device = torch.device("cpu")
        if useGPU and torch.cuda.is_available():
            self.device = torch.device("cuda")

        self.recommended = recommended
        self.featuring_model = featuring_model
        self.featuring_model.changeDevice(useGPU)

        self.cosine_similarity_model = lambda x, y: (F.cosine_similarity(x, y, dim=0)+1)/2
        self.l1_similarity_model = lambda x, y: F.tanh(1/(F.l1_loss(x, y) + (1e-8)))

        self.personal_color_type = list(PERSONAL_COLOR_RGB.keys())

        self.alpha = alpha


    def getPersonalColor(self, user_input_features):
        print(user_input_features[0]["skin"].keys())
        average_rgb = user_input_features[0]["skin"][AVERAGE_RGB]
        if len(user_input_features) > 1:
            for feature in user_input_features[1:]:
                average_rgb += feature["skin"][AVERAGE_RGB]
            average_rgb /= len(user_input_features)
        average_rgb = average_rgb.tolist()

        average_hsv = rgb_to_hsv(*average_rgb)
        H = float(average_hsv[0])
        S = float(average_hsv[1])
        V = float(average_hsv[2])
        diff = round(V - S, 2)

        ans = -1
        if H >= 23 and H <= 203:
            if diff >= 46.25:
                if S >= 31.00:
                    ans = 0
                    # Warm Spring Bright
                else:
                    ans = 1
                    # Warm Spring Light

            elif diff < 46.25:
                if S >= 46.22:
                    ans = 2
                    # Warm Autumn Deep
                else:
                    ans = 3
                    # Warm Autumn Mute

        elif (H >= 0 and H < 23) or (H > 203 and H <= 360):
            if diff >= 48.75:
                if diff >= 28.47:
                    ans = 4
                    # Cool Summer Light
                else:
                    ans = 5
                    # Cool Summer Mute

            elif diff < 48.75:
                if diff >= 31.26:
                    ans = 6
                    # Cool Winter Bright
                else:
                    ans = 7
                    # Cool Winter Deep

        else:
            ans = -1
            # 에러

        return self.personal_color_type[ans]


    def getSimilarity(self, user_feature, target_input, type: str, personal_color: str):
        target_feature = torch.load(target_input, map_location=self.device)

        last_activation_volume_similarity = self.cosine_similarity_model(
            torch.flatten(user_feature[type][LAST_ACTIVATION_VOLUME]),
            torch.flatten(target_feature[type][LAST_ACTIVATION_VOLUME]))

        gram_matrix_similarity = self.l1_similarity_model(user_feature[type][GRAM_MATRIX],
                                                          target_feature[type][GRAM_MATRIX])

        personal_color_rgb = PERSONAL_COLOR_RGB[personal_color] + (LOWER_DEFAULT_COLOR if type=="lower" else [])
        personal_color_similarity = self.l1_similarity_model(target_feature[type][AVERAGE_RGB], torch.tensor(personal_color_rgb[0]))
        for rgb in personal_color_rgb[1:]:
            new_sim = self.l1_similarity_model(target_feature[type][AVERAGE_RGB], torch.tensor(rgb))
            personal_color_similarity = max(personal_color_similarity, new_sim)

        final_similarity = last_activation_volume_similarity * self.alpha[0] + gram_matrix_similarity * self.alpha[1] + personal_color_similarity * self.alpha[2]
        final_similarity /= sum(self.alpha)

        return final_similarity


    def __call__(self,
                 user_inputs: List[Image.Image],
                 type: Literal["upper", "lower"],
                 k: int = 5,
                 personal_color: Optional[str] = None):
        '''
        유저 입력을 토대로 유사도를 계산하여 상위 k개를 반환하는 함수입니다.

        :param user_inputs: Pillow의 Image 객체로 구성된 List. 유저의 사진이 업로드 되어야 한다.
        :param type: 추천받고자 하는 부분이 상의라면 "upper", 하의라면 "lower"
        :param personal_color: 퍼스널 컬러를 유저가 입력했다면 해당 색상을 넘겨주고, 없다면 None으로 놔두면 자동으로 퍼스널 컬러를 추출해준다.
                                반드시 PERSONAL_COLOR_RGB의 key 중 하나로 입력이 되어야 한다.
        :param k: 유사도 기준 상위 몇개의 이미지를 반환할 것인지

        :return: 유사도 기준 상위 k개의 이미지 파일 경로, 추천 대상 상품들의 유사도 배열
        '''
        user_features = [self.featuring_model(user_input) for user_input in user_inputs]

        if personal_color==None:
            personal_color = self.getPersonalColor(user_features)

        similarity_result = {path:0.0 for path in self.recommended[type]}

        for user_feature in user_features:
            for target_input in self.recommended[type]:
                similarity_result[target_input] += self.getSimilarity(user_feature, target_input, type, personal_color).to(self.cpu_device).item()

        sorted_similarity_result = sorted(similarity_result.items(), key=lambda item: item[1], reverse=True)

        return [k for k, v in sorted_similarity_result][:k], sorted_similarity_result


if __name__=="__main__":
    test_recommended = {
        "upper":[f"../test/features/feature{i}.pt" for i in range(0, 14+1)],
        "lower":[],
    }
    model = SimilarityModel(test_recommended, FeaturingModel())

    user_inputs = []
    for i in range(15, 20+1):
        user_inputs.append(Image.open(f"../test/image/{i}.jpg").convert("RGB"))

    topk, similarity_result = model(user_inputs, "upper")
    print(topk)
    print("\n")
    print(similarity_result)

