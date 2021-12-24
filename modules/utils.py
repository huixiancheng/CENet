import torch.nn as nn
import torch
from torch.nn import functional as F



class CAM(nn.Module):
  def __init__(self, inplanes):
    super(CAM, self).__init__()
    self.inplanes = inplanes
    self.pool = nn.MaxPool2d(7, 1, 3)
    self.squeeze = nn.Conv2d(inplanes, inplanes // 16,
                             kernel_size=1, stride=1)
    self.squeeze_bn = nn.BatchNorm2d(inplanes // 16)
    self.relu = nn.ReLU(inplace=True)
    self.unsqueeze = nn.Conv2d(inplanes // 16, inplanes,
                               kernel_size=1, stride=1)
    self.unsqueeze_bn = nn.BatchNorm2d(inplanes)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    # 7x7 pooling
    y = self.pool(x)
    # squeezing and relu
    y = self.relu(self.squeeze_bn(self.squeeze(y)))
    # unsqueezing
    y = self.sigmoid(self.unsqueeze_bn(self.unsqueeze(y)))
    # attention
    return y * x

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

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode='bilinear'):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        print(self.scale_factor, self.mode)
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)
        return x

class Aggregation(nn.Module):
    def __init__(self, channel):
        super(Aggregation, self).__init__()
        self.relu = nn.LeakyReLU()

#         self.upsample2 = Interpolate(scale_factor=2)
#         self.upsample4 = Interpolate(scale_factor=4)
#         self.upsample8 = Interpolate(scale_factor=8)

        self.upsample2 = Interpolate(scale_factor=2)
        self.upsample4 = nn.Sequential(
                            Interpolate(scale_factor=2),
                            Interpolate(scale_factor=2),)
        self.upsample8 = nn.Sequential(
                            Interpolate(scale_factor=2),
                            Interpolate(scale_factor=2),
                            Interpolate(scale_factor=2),)

        self.conv_upsample1_2 = BasicConv2d(channel, channel, 3, padding=1, relu=False)
        self.conv_upsample1_3 = BasicConv2d(channel, channel, 3, padding=1, relu=False)
        self.conv_upsample1_4 = BasicConv2d(channel, channel, 3, padding=1, relu=False)
        self.conv_upsample2_3 = BasicConv2d(channel, channel, 3, padding=1, relu=False)
        self.conv_upsample2_4 = BasicConv2d(channel, channel, 3, padding=1, relu=False)
        self.conv_upsample3_4 = BasicConv2d(channel, channel, 3, padding=1, relu=False)

#         self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
#         self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
#         self.conv_concat4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)

#         self.conv4 = BasicConv2d(4*channel, 4*channel, 1)


    def forward(self, x1, x2, x3, x4):
        x1_1 = x1
        x2_1 = self.relu(self.conv_upsample1_2(self.upsample2(x1)) + x2)
        x3_1 = self.relu(self.conv_upsample1_3(self.upsample4(x1)) + self.conv_upsample2_3(self.upsample2(x2)) + x3)
        x4_1 = self.relu(self.conv_upsample1_4(self.upsample8(x1)) + self.conv_upsample2_4(self.upsample4(x2)) \
               + self.conv_upsample3_4(self.upsample2(x3)) + x4)

#         x2_2 = torch.cat((x2_1, self.upsample2(x1_1)), 1)
#         x2_2 = self.conv_concat2(x2_2)

#         x3_2 = torch.cat((x3_1, self.upsample2(x2_2)), 1)
#         x3_2 = self.conv_concat3(x3_2)

#         x4_2 = torch.cat((x4_1, self.upsample2(x3_2)), 1)
#         x4_2 = self.conv_concat4(x4_2)

#         x = self.conv4(x4_2)

        return x1_1, x2_1, x3_1, x4_1

# class aggregation(nn.Module):
#     # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
#     def __init__(self, channel):
#         super(aggregation, self).__init__()

#         self.block_1 = nn.Sequential(
#             BasicConv2d(channel, channel, 3, padding=1),
#             Interpolate(scale_factor=2),
#             BasicConv2d(channel, channel, 3, padding=1),
#         )

#         self.block_2 = nn.Sequential(
#             BasicConv2d(channel, channel, 3, padding=1),
#             Interpolate(scale_factor=2),
#             BasicConv2d(channel, channel, 3, padding=1),
#         )

#         self.block_3 = nn.Sequential(
#             BasicConv2d(channel, channel, 3, padding=1),
#             Interpolate(scale_factor=2),
#             BasicConv2d(channel, channel, 3, padding=1),
#         )

#         self.block_4 = nn.Sequential(
#             BasicConv2d(channel, channel, 3, padding=1),
#             Interpolate(scale_factor=2),
#             BasicConv2d(channel, channel, 3, padding=1),
#         )

#         self.block_5 = nn.Sequential(
#             BasicConv2d(channel, channel, 3, padding=1),
#             Interpolate(scale_factor=2),
#             BasicConv2d(channel, channel, 3, padding=1),
#         )
#         self.block_6 = nn.Sequential(
#             BasicConv2d(channel, channel, 3, padding=1),
#             Interpolate(scale_factor=2),
#             BasicConv2d(channel, channel, 3, padding=1),
#         )
#         self.block_7 = nn.Sequential(
#             BasicConv2d(channel, channel, 3, padding=1),
#             Interpolate(scale_factor=2),
#             BasicConv2d(channel, channel, 3, padding=1),
#         )
#         self.block_8 = nn.Sequential(
#             BasicConv2d(channel, channel, 3, padding=1),
#             Interpolate(scale_factor=4),
#             BasicConv2d(channel, channel, 3, padding=1),
#         )
#         self.block_9 = nn.Sequential(
#             BasicConv2d(channel, channel, 3, padding=1),
#             Interpolate(scale_factor=8),
#             BasicConv2d(channel, channel, 3, padding=1),
#         )

#     def forward(self, x1, x2, x3, x4):
#         x_22 = self.block_1(x1) + x2
#         x_23 = self.block_2(x2) + x3
#         x_24 = self.block_3(x3) + x4

#         x_33 = self.block_4(x_22) + x_23 + x3
#         x_34 = self.block_5(x_23) + x_24

#         x_44 = self.block_6(x_33) + x_34 + x4

#         x_43 = self.block_7(x_33) + x_44
#         x_42 = self.block_8(x_22) + x_43
#         x_41 = self.block_9(x1) + x_42

#         return x_41

class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    def __init__(self, channel):
        super(aggregation, self).__init__()

        self.upsample2 = Interpolate(scale_factor=2)
        self.upsample4 = Interpolate(scale_factor=4)
        self.upsample8 = Interpolate(scale_factor=8)

        self.block_1 = nn.Sequential(
            BasicConv2d(2*channel, channel, 1),
            BasicConv2d(channel, channel, 3, padding=1),
        )

        self.block_2 = nn.Sequential(
            BasicConv2d(2*channel, channel, 1),
            BasicConv2d(channel, channel, 3, padding=1),
        )

        self.block_3 = nn.Sequential(
            BasicConv2d(2*channel, channel, 1),
            BasicConv2d(channel, channel, 3, padding=1),
        )

        self.block_4 = nn.Sequential(
            BasicConv2d(3*channel, channel, 1),
            BasicConv2d(channel, channel, 3, padding=1),
        )

        self.block_5 = nn.Sequential(
            nn.Conv2d(2*channel, channel, 1),
            BasicConv2d(channel, channel, 3, padding=1),
        )
        self.block_6 = nn.Sequential(
            nn.Conv2d(3*channel, channel, 1),
            BasicConv2d(channel, channel, 3, padding=1),
        )
        self.block_7 = nn.Sequential(
            nn.Conv2d(2*channel, channel, 1),
            BasicConv2d(channel, channel, 3, padding=1),
        )
        self.block_8 = nn.Sequential(
            nn.Conv2d(2*channel, channel, 1),
            BasicConv2d(channel, channel, 3, padding=1),
        )
        self.block_9 = nn.Sequential(
            nn.Conv2d(2*channel, channel, 1),
            BasicConv2d(channel, channel, 3, padding=1),
        )

    def forward(self, x1, x2, x3, x4):
        x_22 = torch.cat((self.upsample2(x1), x2), 1)
        x_22 = self.block_1(x_22)

        x_23 = torch.cat((self.upsample2(x2), x3), 1)
        x_23 = self.block_2(x_23)

        x_24 = torch.cat((self.upsample2(x3), x4), 1)
        x_24 = self.block_3(x_24)

        x_33 = torch.cat((self.upsample2(x_22), x_23, x3), 1)
        x_33 = self.block_4(x_33)

        x_34 = torch.cat((self.upsample2(x_23), x_24), 1)
        x_34 = self.block_5(x_34)

        x_44 = torch.cat((self.upsample2(x_33), x_34, x4), 1)
        x_44 = self.block_6(x_44)

        x_43 = torch.cat((self.upsample2(x_33), x_44), 1)
        x_43 = self.block_7(x_43)

        x_42 = torch.cat((self.upsample4(x_22), x_43), 1)
        x_42 = self.block_8(x_42)

        x_41 = torch.cat((self.upsample8(x1), x_42), 1)
        x_41 = self.block_9(x_41)

        return x_22, x_33, x_41

class PyramBranch(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(PyramBranch, self).__init__()
        if padding == 0:
            print("Not supported for conv 1x1")

        else:
#             self.atrous_conv3x1 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1),
#                                                 nn.Conv2d(planes, planes, kernel_size=(kernel_size, 1),
#                                                 stride=1, padding=(padding, 0), dilation=(dilation, 1), bias=False))
            
            self.atrous_conv3x1 = nn.Conv2d(inplanes, planes, kernel_size=(kernel_size, 1),
                                            stride=1, padding=(padding, 0), dilation=(dilation, 1), bias=False)

            self.atrous_conv1x3 = nn.Conv2d(planes, planes, kernel_size=(1, kernel_size),
                                            stride=1, padding=(0, padding), dilation=(1, dilation), bias=False)

            self.bn3x1 = nn.BatchNorm2d(planes)
            self.relu3x1 = nn.LeakyReLU()

            self.bn1x3 = nn.BatchNorm2d(planes)
            self.relu1x3 = nn.LeakyReLU()

    def forward(self, x):
        x = self.atrous_conv3x1(x)
        x = self.bn3x1(x)
        x = self.relu3x1(x)

        x = self.atrous_conv1x3(x)
        x = self.bn1x3(x)

        return self.relu1x3(x)

class DAPF(nn.Module):
    def __init__(self, inplanes, outplanes, alpha=4):
        super(DAPF, self).__init__()

        dilations = [1, 3, 5, 7]
        mid_planes = inplanes // alpha
        self.conv1x1 = nn.Conv2d(inplanes, mid_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1x1 = nn.BatchNorm2d(mid_planes)
        self.relu1x1 = nn.LeakyReLU()

        self.pyBranch2 = PyramBranch(inplanes, mid_planes, 3, padding=dilations[1], dilation=dilations[1])
        self.pyBranch3 = PyramBranch(inplanes, mid_planes, 3, padding=dilations[2], dilation=dilations[2])
        self.pyBranch4 = PyramBranch(inplanes, mid_planes, 3, padding=dilations[3], dilation=dilations[3])

        self.conv1 = nn.Conv2d(mid_planes*4, outplanes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)

        self.conv_res = nn.Conv2d(inplanes, outplanes, 1, bias=False)
        self.bn_res = nn.BatchNorm2d(outplanes)

        self.relu = nn.LeakyReLU()


    def forward(self, x):
        x1 = self.conv1x1(x)
        x1 = self.bn1x1(x1)
        x1 = self.relu1x1(x1)

        x2 = self.pyBranch2(x)
        x3 = self.pyBranch3(x)
        x4 = self.pyBranch4(x)

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        x_cat = self.conv1(x_cat)
        x_cat = self.bn1(x_cat)

        x = self.relu(x_cat + self.bn_res(self.conv_res(x)))

        return x

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel, map_reduce=4):
        super(RFB_modified, self).__init__()
        inter_channel = in_channel // map_reduce

        self.relu = nn.LeakyReLU()
        dilations = [3, 5, 7]
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, inter_channel, 1, relu=False),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, inter_channel, 1, relu=False),
            BasicConv2d(inter_channel, inter_channel, kernel_size=(1, 3), padding=(0, 1), relu=False),
            BasicConv2d(inter_channel, inter_channel, kernel_size=(3, 1), padding=(1, 0), relu=False),
            BasicConv2d(inter_channel, inter_channel, 3, padding=dilations[0], dilation=dilations[0], relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, inter_channel, 1, relu=False),
            BasicConv2d(inter_channel, inter_channel, kernel_size=(1, 5), padding=(0, 2), relu=False),
            BasicConv2d(inter_channel, inter_channel, kernel_size=(5, 1), padding=(2, 0), relu=False),
            BasicConv2d(inter_channel, inter_channel, 3, padding=dilations[1], dilation=dilations[1], relu=False)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, inter_channel, 1, relu=False),
            BasicConv2d(inter_channel, inter_channel, kernel_size=(1, 7), padding=(0, 3), relu=False),
            BasicConv2d(inter_channel, inter_channel, kernel_size=(7, 1), padding=(3, 0), relu=False),
            BasicConv2d(inter_channel, inter_channel, 3, padding=dilations[2], dilation=dilations[2], relu=False)
        )
        self.conv_cat = BasicConv2d(4 * inter_channel, out_channel, 1, relu=False)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1, relu=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))
        return x


# class RFB_modified(nn.Module):

#     def __init__(self, in_planes, out_planes, stride=1, scale=1, map_reduce=8, vision=1):
#         super(RFB_modified, self).__init__()
#         self.scale = scale
#         self.out_channels = out_planes
#         inter_planes = in_planes // map_reduce

#         self.branch0 = nn.Sequential(
#             BasicConv2d(in_planes, inter_planes, kernel_size=1, stride=1, relu=False),
#             BasicConv2d(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
#             BasicConv2d(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 1, dilation=vision + 1, relu=False)
#         )
#         self.branch1 = nn.Sequential(
#             BasicConv2d(in_planes, inter_planes, kernel_size=1, stride=1, relu=False),
#             BasicConv2d(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
#             BasicConv2d(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 2, dilation=vision + 2, relu=False)
#         )
#         self.branch2 = nn.Sequential(
#             BasicConv2d(in_planes, inter_planes, kernel_size=1, stride=1, relu=False),
#             BasicConv2d(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),
#             BasicConv2d((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
#             BasicConv2d(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 4, dilation=vision + 4, relu=False)
#         )

#         self.ConvLinear = BasicConv2d(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
#         self.shortcut = BasicConv2d(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
#         self.relu = nn.LeakyReLU()

#     def forward(self, x):
#         x0 = self.branch0(x)
#         x1 = self.branch1(x)
#         x2 = self.branch2(x)

#         out = torch.cat((x0, x1, x2), 1)
#         out = self.ConvLinear(out)
#         short = self.shortcut(x)
#         out = out * self.scale + short
#         out = self.relu(out)

#         return out

from torch.nn.parameter import Parameter

class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=16):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out

class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(ResContextBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1))
        self.act1 = nn.LeakyReLU()

        self.conv2 = BasicConv2d(out_filters, out_filters, 3, padding=1)
        self.conv3 = BasicConv2d(out_filters, out_filters, 3, padding=2, dilation=2)


    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(shortcut)
        resA = self.conv3(resA)

        output = shortcut + resA

        return output


class CTM(nn.Module):
    def __init__(self, C1=3, C2=8, C3=16, C4=32):
        super(CTM, self).__init__()
        self.conva = ResContextBlock(C1, C2)
        self.convb = ResContextBlock(C2, C3)
        self.convc = ResContextBlock(C3, C4)
        self.sca = sa_layer(C4)

    def forward(self, x):
        x = self.conva(x)
        x = self.convb(x)
        x = self.convc(x)
        x = self.sca(x)
        return x

class Propose(nn.Module):
    def __init__(self):
        super(Propose, self).__init__()
        self.conv_range = CTM(1, 8, 16, 32)
        self.conv_xyz = CTM(3, 8, 16, 32)
        self.conv_remission = CTM(1, 8, 16, 32)
#         self.conv_normal = CTM(3, 32)

        self.conv0 = nn.Conv2d(96, 64, kernel_size=(1, 1))
#         self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.pr = nn.LeakyReLU()
        self.sca = sa_layer(channel=64)

    def forward(self, xyzdr):
        ranges = self.conv_range(xyzdr[:, 0, None, :, :])
        xyz = self.conv_xyz(xyzdr[:, 1:4, :, :])
        remission = self.conv_remission(xyzdr[:, 4, None, :, :])

#         normal = self.conv_normal(xyzdr[:, 5:, :, :])

#         feature = torch.cat((xyz, remission, ranges, normal), dim=1)
        feature = torch.cat((ranges, xyz, remission), dim=1)

        feature = self.conv0(feature)
#         feature = self.conv1(feature)
        feature = self.bn(feature)
        feature = self.sca(feature)
        feature = self.pr(feature)
        return feature




# class CTM(nn.Module):
#     def __init__(self, C1=3, C2=8, C3=16, C4=32):
#         super(CTM, self).__init__()
#         self.conva = ResContextBlock(C1, C2)
#         self.convb = ResContextBlock(C2, C3)
#         self.convc = ResContextBlock(C3, C4)
#         self.sca = sa_layer(C4)

#     def forward(self, x):
#         x = self.conva(x)
#         x = self.convb(x)
#         x = self.convc(x)
#         x = self.sca(x)
#         return x

# class Propose(nn.Module):
#     def __init__(self):
#         super(Propose, self).__init__()
#         self.conv_range = CTM(1, 8, 16, 32)
#         self.conv_xyz = CTM(3, 8, 16, 32)
#         self.conv_remission = CTM(1, 8, 16, 32)