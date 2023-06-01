import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np


# load Mask2Former fine-tuned on Cityscapes instance segmentation
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-cityscapes-instance")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-cityscapes-instance")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

## INPUT SIZE = 1x5x64x512 (1024 or 2048)

class ExtractFeatureBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(5, 3, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(5, 3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(5, 3, kernel_size=5, padding=2)
        self.last_conv = nn.Conv2d(3, 3, kernel_size=1, padding=0)
        self.batchnorm = nn.BatchNorm2d(3)
        self.activation = nn.ReLU()

    def forward(self, x):
        x1, x2, x3 = self.conv1(x), self.conv2(x), self.conv3(x)
        x1, x2, x3 = self.batchnorm(x1), self.batchnorm(x2), self.batchnorm(x3)
        x1, x2, x3 = self.activation(x1), self.activation(x2), self.activation(x3)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.last_conv(x)
        x = self.batchnorm(x)
        x = self.activation(x)



class Mask2Former(nn.Module):
    def __init__(self, nclasses, aux):
        super(Mask2Former, self).__init__()
        self.aux = aux
        self.nclasses = nclasses
        self.feature_extractor = ExtractFeatureBlock()
        self.processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-cityscapes-instance")
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-cityscapes-instance")

    def forward(self, x):
        x = self.feature_extractor(x)
        x = list(x)
        x = self.processor(images=x, return_tensors="pt")
        x = self.model(**x)

        
        return out