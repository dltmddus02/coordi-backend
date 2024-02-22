import torch
import torchvision
import os
from django.conf import settings
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from torchvision.transforms.functional import to_pil_image, crop
import torch.nn as nn
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from typing import *
from copy import deepcopy
from tqdm import tqdm
from .Unnormalize import UnNormalize
DIR = settings.BASE_DIR


CLASS_NUM = 20

CLOTHES_PART_LABEL = {
    "upper":[4, 7],
    "lower":[5, 6],
    #"shoes":[9, 10],
    #"hat":[1],
}

BODY_PART_LABEL = {
    #"hair": [2],
    "skin": [11, 12, 13, 14, 15],
}

COLOR_SPACE_MAP = {
    'GrayScale':1,
    'RGB':3,
    'RGBA':4
}

def save_feature(feature, path):
    torch.save(feature, path)


class ClothClassificationModel(nn.Module):
    def __init__(self, num_classes=CLASS_NUM, last_channel=1280, compressor_channel=320):
        super(ClothClassificationModel, self).__init__()
        backbone = models.efficientnet_b0(pretrained=True)

        self.features = backbone.features
        self.compressor = lambda x: x
        if last_channel != compressor_channel:
            self.compressor = nn.Conv2d(last_channel, compressor_channel, 1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(compressor_channel, num_classes)



    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = self.compressor(x)
        # print(x.shape)

        x = self.avgpool(x)
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)

        x = self.classifier(x)
        return x


class FeaturingModel:
    def __init__(self,
                 useGPU: bool = False,
                 segformer_path: str = "mattmdjaga/segformer_b2_clothes",
                 classifier_path: str = os.path.join(DIR, "outfit/pipeline/checkpoint/classifier_efficientnetb0.pt"),
                 classifier_input_size: int = 224,
                 layer_gram_matrix: int = 4
                 ):
        '''
        특징 추출 모델
        
        :param useGPU: 연산에 GPU를 이용할 것인가. False라면 CPU만 이용, True라면 GPU를 메인으로 이용.
        :param segformer_path: 의상 분할 모델의 경로 (건드리지 말것)
        :param classifier_path: 의상 분류 모델의 경로 (경로 이상하면 수정)
        :param classifier_input_size: 의상 분류 모델의 학습할 때 적용했던 Image Size (건드리지 말것)
        '''
        self.cpu_device = torch.device("cpu")
        self.device = torch.device("cpu")
        if useGPU and torch.cuda.is_available():
            self.device = torch.device("cuda")


        self.segformer_processor = SegformerImageProcessor.from_pretrained(segformer_path)
        self.segformer_model = AutoModelForSemanticSegmentation.from_pretrained(segformer_path)

        self.classifier_model = torch.load(classifier_path, map_location=self.device)
        self.classifier_compressor = self.classifier_model.compressor
        self.classifier_model = self.classifier_model.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier_input_size = classifier_input_size

        self.topilimage = torchvision.transforms.ToPILImage()
        self.totensor = torchvision.transforms.ToTensor()

        self.transformer = transforms.Compose([transforms.ToTensor(),
                                          torchvision.transforms.Resize((self.classifier_input_size, self.classifier_input_size)),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                          ])
        self.unnormalize = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.layer_gram_matrix = layer_gram_matrix

    def changeDevice(self, useGPU: bool):
        self.device = torch.device("cpu")
        if useGPU and torch.cuda.is_available():
            self.device = torch.device("cuda")

    def getPart(self,
                image_ori: Image.Image,
                seg: torch.tensor,
                labels: List[int],
                image_channel: int = 3,
                background: int = 0):
        mask = seg==labels[0]
        if len(labels)>=2:
            for label in labels[1:]:
                mask = torch.logical_or(mask, seg==label)
        masks = torch.stack([mask]*image_channel, dim=0)
        isExist = not torch.all(~masks).item()

        image = self.totensor(image_ori)
        if isExist:
            y, x = torch.where(mask)

            image[~masks] = background

            h = torch.max(y)-torch.min(y)
            w = torch.max(x)-torch.min(x)
            if h>0 and w>0:
                image = self.topilimage(crop(image, torch.min(y), torch.min(x), h, w))
            else:
                image = self.topilimage(torch.zeros_like(image))
        else:
            image = self.topilimage(torch.zeros_like(image))

        return image, masks

    def gram_matrix(self, x):
        n, c, h, w = x.size()
        x = x.view(n * c, h * w)
        gram = torch.mm(x, x.t())
        return gram

    def __call__(self, x, color_space: str = "RGB", only_clothes: bool = False):
        image = x.convert(color_space)

        # 전신 사진 분할
        inputs = self.segformer_processor(images=image, return_tensors="pt")
        outputs = self.segformer_model(**inputs)
        logits = outputs.logits.cpu()
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0]

        # 분할 사진 Feature 추출 - 옷
        result = {}
        PART_LABEL = deepcopy(CLOTHES_PART_LABEL)
        if not only_clothes:
            PART_LABEL.update(BODY_PART_LABEL)

        for name, labels in PART_LABEL.items():
            features = {}
            part_image = self.getPart(image, pred_seg, labels, image_channel=COLOR_SPACE_MAP[color_space])[0]

            input_classifier = self.transformer(part_image).unsqueeze(0).to(self.device)
            activation_volume = self.classifier_model[:self.layer_gram_matrix](input_classifier)
            output_classifier = self.classifier_model[self.layer_gram_matrix:](activation_volume)
            output_classifier = self.classifier_compressor(output_classifier)

            features["last_activation_volume"] = torch.flatten(self.avgpool(output_classifier), 1).squeeze(0).to(self.cpu_device)
            features["gram_matrix"] = self.gram_matrix(activation_volume).to(self.cpu_device)
            features["average_rgb"] = 255*self.unnormalize(input_classifier).squeeze(0).mean(dim=-1).mean(dim=-1)

            result[name] = features

        return result

if __name__=="__main__":
    model = FeaturingModel()
    for i in tqdm(range(0, 0+1)):
        feature = model(Image.open(f"../data/slowand/image/{i}.jpg"), only_clothes=True)
        save_feature(feature, f"../data/slowand/features/{i}.pt")
    print("Complete")
