import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from transformers import Mask2FormerModel, AutoFeatureExtractor


# load Mask2Former fine-tuned on Cityscapes instance segmentation
#processor = AutoFeatureExtractor.from_pretrained("facebook/mask2former-swin-small-cityscapes-instance")
#model = Mask2FormerModel.from_pretrained("facebook/mask2former-swin-small-cityscapes-instance")



processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-cityscapes-instance")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-cityscapes-instance")

# for param in model.parameters():
#     print(param)
#     param.requires_grad = False
    
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

#pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
## CREATE AN INPUT USING TORCH RANDN FROM 0 TO 1: 1x3x64x2048

images = torch.rand(4, 3, 64, 2048).cuda()

image = list(images)
test = Mask2FormerModel.from_pretrained("facebook/mask2former-swin-small-cityscapes-instance")
#image = [torch.rand(3, 64, 2048),torch.rand(3, 64, 2048), torch.rand(3, 64, 2048)]
inputs = processor(images=image, return_tensors="pt")
a = test.dummy_inputs

with torch.no_grad():
    outputs = model(**inputs,output_hidden_states=True)

print(model)

