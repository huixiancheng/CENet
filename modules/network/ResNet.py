import sys
sys.path.append('/home/son/project/pham_wrapper/CENet')
print(sys.path)
import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights, deeplabv3_resnet50
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from third_party.SwinFusion.models.network_swinfusion import SwinFusion as net
from torchvision.transforms.transforms import Resize


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=True):
        super(BasicConv2d, self).__init__()
        self.relu = relu
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        if self.relu:
            self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x

class Final_Model(nn.Module):

    def __init__(self, backbone_net, semantic_head):
        super(Final_Model, self).__init__()
        self.backend = backbone_net
        self.semantic_head = semantic_head

    def forward(self, x):
        middle_feature_maps = self.backend(x)

        semantic_output = self.semantic_head(middle_feature_maps)

        return semantic_output


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, if_BN=None):
        super(BasicBlock, self).__init__()
        self.if_BN = if_BN
        if self.if_BN:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        if self.if_BN:
            self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU()
        self.conv2 = conv3x3(planes, planes)
        if self.if_BN:
            self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.if_BN:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.if_BN:
            out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet_34(nn.Module):
    def __init__(self, nclasses, aux=False, block=BasicBlock, layers=[3, 4, 6, 3], if_BN=True, zero_init_residual=False,
                 norm_layer=None, groups=1, width_per_group=64):
        super(ResNet_34, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.if_BN = if_BN
        self.dilation = 1
        self.aux = aux

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = BasicConv2d(5, 64, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = BasicConv2d(128, 128, kernel_size=3, padding=1)

        self.inplanes = 128

        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)

        self.conv_1 = BasicConv2d(640, 256, kernel_size=3, padding=1)
        self.conv_2 = BasicConv2d(256, 128, kernel_size=3, padding=1)
        self.semantic_output = nn.Conv2d(128, nclasses, 1)

        if self.aux:
            self.aux_head1 = nn.Conv2d(128, nclasses, 1)
            self.aux_head2 = nn.Conv2d(128, nclasses, 1)
            self.aux_head3 = nn.Conv2d(128, nclasses, 1)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.if_BN:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, if_BN=self.if_BN))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                if_BN=self.if_BN))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_1 = self.layer1(x)  # 1
        x_2 = self.layer2(x_1)  # 1/2
        x_3 = self.layer3(x_2)  # 1/4
        x_4 = self.layer4(x_3)  # 1/8

        res_2 = F.interpolate(x_2, size=x.size()[2:], mode='bilinear', align_corners=True)
        res_3 = F.interpolate(x_3, size=x.size()[2:], mode='bilinear', align_corners=True)
        res_4 = F.interpolate(x_4, size=x.size()[2:], mode='bilinear', align_corners=True)
        res = [x, x_1, res_2, res_3, res_4]

        out = torch.cat(res, dim=1)
        out = self.conv_1(out)
        out = self.conv_2(out)
        out = self.semantic_output(out)
        out = F.softmax(out, dim=1)

        if self.aux:
            res_2 = self.aux_head1(res_2)
            res_2 = F.softmax(res_2, dim=1)

            res_3 = self.aux_head2(res_3)
            res_3 = F.softmax(res_3, dim=1)

            res_4 = self.aux_head3(res_4)
            res_4 = F.softmax(res_4, dim=1)

        if self.aux:
            return [out, res_2, res_3, res_4]
        else:
            return out
        
class Fusion(nn.Module):
    def __init__(self,nclasses, aux = True) -> None:
        super().__init__()
        self.num_classes = nclasses
        self.aux = aux
        self.backbone  = resnet_fpn_backbone('resnet50', weights='ResNet50_Weights.IMAGENET1K_V2', trainable_layers=5) 
        self.feature_extract = FeatureExtractionBlock()
        self.init_conv = BasicConv2d(5, 3, kernel_size=3, padding=1)
        self.feature_reduction_1 =BasicConv2d(256, 128, kernel_size=1, padding=0)
        self.feature_reduction_2 =BasicConv2d(128, 64, kernel_size=1, padding=0)
        self.predict_com =BasicConv2d(64*4, self.num_classes, kernel_size=1, padding=0)
        self.predict =BasicConv2d(64, self.num_classes, kernel_size=1, padding=0)

    def forward(self, x, rgb):
        x_feature = self.feature_extract(x)
        x = self.init_conv(x)
        x = self.backbone(x)
        x.pop("pool")
        x.pop("3")
        x = list(x.values())
        x = [x_feature] + x
        for i, feature in enumerate(x):
            x[i] = self.feature_reduction_2(self.feature_reduction_1(F.interpolate(feature, size=x_feature.size()[2:], mode='bilinear', align_corners=True)))

        final_predict = self.predict_com(torch.cat(x, dim=1))
        final_predict = F.softmax(final_predict, dim=1)
        if self.aux:
            predict_1 = F.softmax(self.predict(x[1]), dim=1)
            predict_2 = F.softmax(self.predict(x[2]), dim=1)
            predict_3 = F.softmax(self.predict(x[3]), dim=1)
            return [final_predict, predict_1, predict_2, predict_3]
        else:
            return final_predict






        return x
        
# class Fusion(nn.Module):
#     def __init__(self, nclasses, aux = True, block=BasicBlock, layers=[3, 4, 6, 3], if_BN=True, zero_init_residual=False,
#                  norm_layer=None, groups=1, width_per_group=64):
#         super(Fusion, self).__init__()
#         self.num_classes = nclasses
#         self.aux = aux

#         self.conv3_1 = BasicConv2d(5, 3, kernel_size=3, padding=1)

#         self.model = deeplabv3_resnet50(weights = None, num_classes = nclasses, aux_loss = True, weights_backbone = "ResNet50_Weights.IMAGENET1K_V2")
#         print(self.model)

#     def forward(self, x, rgb):

#         x = self.conv3_1(x)
#         out = self.model(x)
#         if self.aux:
    
#             return [F.softmax(out["out"],dim=1), F.softmax(out["aux"],dim=1)]
#         else:

#             return F.softmax(out["out"],dim=1)
        



# class Fusion(nn.Module):
#     def __init__(self, nclasses, aux = True, block=BasicBlock, layers=[3, 4, 6, 3], if_BN=True, zero_init_residual=False,
#                  norm_layer=None, groups=1, width_per_group=64):
#         super(Fusion, self).__init__()
#         self.num_classes = nclasses
#         self.aux = aux

#         self.conv3_1 = BasicConv2d(5, 3, kernel_size=3, padding=1)
#         self.conv1 = BasicConv2d(256, 256, kernel_size=1, padding=0)
#         self.conv1_0 = BasicConv2d(256, 256, kernel_size=1, padding=0)
#         self.conv1_1 = BasicConv2d(256, 256, kernel_size=1, padding=0)
#         self.conv1_2 = BasicConv2d(256, 256, kernel_size=1, padding=0)
#         self.conv1_3 = BasicConv2d(256, 256, kernel_size=1, padding=0)
#         self.feature_extractor = FeatureExtractionBlock()
#         self.semantic_output = nn.Conv2d(256, nclasses, 1)
#         #self.model = deeplabv3_resnet50(weights = None, num_classes = nclasses, aux_loss = True, weights_backbone = "ResNet50_Weights.IMAGENET1K_V2")
#         self.model = fasterrcnn_resnet50_fpn_v2(weights_backbone = "ResNet50_Weights.IMAGENET1K_V2").backbone
#         print(self.model)

#     def forward(self, x, rgb):
#         x_feature = self.feature_extractor(x)
#         x = self.conv3_1(x)
#         out = self.model(x)

#         out_4 = F.sigmoid(self.semantic_output(x_feature), dim=1)

#         out_3 = self.conv1_3(out["3"])
#         out_2 = self.conv1_2(out["2"])  + F.interpolate(out_3, size=out["2"].size()[2:], mode='bilinear', align_corners=True)
#         out_1 = self.conv1_1(out["1"])  + F.interpolate(out_2, size=out["1"].size()[2:], mode='bilinear', align_corners=True)
#         out_0 = self.conv1_0(out["0"])  + F.interpolate(out_1, size=out["0"].size()[2:], mode='bilinear', align_corners=True)
#         out = self.conv1(x_feature) + F.interpolate(out_0, size=x_feature.size()[2:], mode='bilinear', align_corners=True)


#         if self.aux:
#             out_0 = self.semantic_output(out_0)
#             out_0 = F.interpolate(out_0, size=x.size()[2:], mode='bilinear', align_corners=True)
#             out_0 = F.softmax(out_0,dim=1)

#             out_1 = self.semantic_output(out_1)
#             out_1 = F.interpolate(out_1, size=x.size()[2:], mode='bilinear', align_corners=True)
#             out_1 = F.softmax(out_1,dim=1)

#             out_2 = self.semantic_output(out_2)
#             out_2 = F.interpolate(out_2, size=x.size()[2:], mode='bilinear', align_corners=True)
#             out_2 = F.softmax(out_2,dim=1)

#             out = self.semantic_output(out)
#             out = F.softmax(out,dim=1)

#             return [out, out_0, out_1, out_2]

#         else:
#             out = self.semantic_output(out)
#             return F.softmax(out["out"],dim=1)


       

class FeatureExtractionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(5, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(5, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(5, 128, kernel_size=5, stride=1, padding=2, bias=False)

        self.final_conv = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn_final = nn.BatchNorm2d(256)

        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x1, x2, x3 = self.conv1(x), self.conv2(x), self.conv3(x)
        x1, x2, x3 = self.bn1(x1), self.bn2(x2), self.bn3(x3)
        x1, x2, x3 = self.activation(x1), self.activation(x2), self.activation(x3)
        x = x1+x2+x3
        x = self.final_conv(x)
        x = self.bn_final(x)
        x_feature = self.activation(x)
   

        return x_feature


if __name__ == "__main__":
    import time
    model = Fusion(20).cuda()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: ", pytorch_total_params / 1000000, "M")
    time_train = []
    for i in range(20):
        input_3D = torch.randn(2, 5, 64, 512).cuda()
        input_rgb = torch.randn(2, 3, 452, 1032).cuda()
        model.eval()
        with torch.no_grad():
          start_time = time.time()
          outputs = model(input_3D, input_rgb)
        torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        fwt = time.time() - start_time
        time_train.append(fwt)
        print ("Forward time per img: %.3f (Mean: %.3f)" % (
          fwt / 1, sum(time_train) / len(time_train) / 1))
        time.sleep(0.15)

    # for i in range(20):
    #     inputs = torch.randn(1, 5, 64, 2048).cuda()
    #     model.eval()
    #     with torch.no_grad():
    #       start_time = time.time()
    #       outputs = model(inputs)
    #     torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
    #     fwt = time.time() - start_time
    #     time_train.append(fwt)
    #     print ("Forward time per img: %.3f (Mean: %.3f)" % (
    #       fwt / 1, sum(time_train) / len(time_train) / 1))
    #     time.sleep(0.15)




