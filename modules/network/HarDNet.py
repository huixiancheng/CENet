import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
# from CatConv2d.catconv2d import CatConv2d


class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                                          stride=stride, padding=kernel // 2, bias=False))
        self.add_module('norm', nn.BatchNorm2d(out_channels))
        self.add_module('relu', nn.LeakyReLU(inplace=True))

        # print(kernel, 'x', kernel, 'x', in_channels, 'x', out_channels)

    def forward(self, x):
        return super().forward(x)


class BRLayer(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()

        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.LeakyReLU(True))

    def forward(self, x):
        return super().forward(x)


class HarDBlock_v2(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False, dwconv=False,
                 list_out=False):
        super().__init__()
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.grmul = grmul
        self.n_layers = n_layers
        self.keepBase = keepBase
        self.links = []
        self.list_out = list_out
        layers_ = []
        self.out_channels = 0

        for i in range(n_layers):
            outch, inch, link = self.get_link(i + 1, in_channels, growth_rate, grmul)
            self.links.append(link)
            use_relu = residual_out
            layers_.append(CatConv2d(inch, outch, (3, 3), relu=True))

            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch
        print("Blk out =", self.out_channels)
        self.layers = nn.ModuleList(layers_)

    def transform(self, blk):
        for i in range(len(self.layers)):
            self.layers[i].weight[:, :, :, :] = blk.layers[i][0].weight[:, :, :, :]
            self.layers[i].bias[:] = blk.layers[i][0].bias[:]

    def forward(self, x):
        layers_ = [x]
        # self.res = []
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])

            out = self.layers[layer](tin)
            # self.res.append(out)
            layers_.append(out)
        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keepBase) or \
                    (i == t - 1) or (i % 2 == 1):
                out_.append(layers_[i])
        if self.list_out:
            return out_
        else:
            return torch.cat(out_, 1)


class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False):
        super().__init__()
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.grmul = grmul
        self.n_layers = n_layers
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0  # if upsample else in_channels
        for i in range(n_layers):
            outch, inch, link = self.get_link(i + 1, in_channels, growth_rate, grmul)
            self.links.append(link)
            use_relu = residual_out
            layers_.append(ConvLayer(inch, outch))
            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch
        # print("Blk out =",self.out_channels)
        self.layers = nn.ModuleList(layers_)

    def forward(self, x):
        layers_ = [x]
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)
        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keepBase) or \
                    (i == t - 1) or (i % 2 == 1):
                out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out



class HarDNet(nn.Module):
    def __init__(self, nclasses=20, aux=False):
        super(HarDNet, self).__init__()

#         first_ch = [16, 24, 32, 48]
#         ch_list = [64, 96, 160, 224]
#         grmul = 1.7
#         gr = [10, 16, 18, 24]
#         n_layers = [4, 4, 8, 8]
#         downSamp = [0, 1, 1, 1]
#         self.shortcut_layers = [3, 4, 7, 10]

        first_ch = [64, 128, 128]
        ch_list = [128, 128, 128, 128]
        grmul = 1.6
        gr = [16, 16, 16, 16]
        n_layers = [8, 8, 8, 8]
        downSamp = [0, 1, 1, 1]
        self.shortcut_layers = [2, 3, 6, 9]

        blks = len(n_layers)
        self.aux = aux


        blks = len(n_layers)


        self.base = nn.ModuleList([])
        self.base.append(ConvLayer(in_channels=5, out_channels=first_ch[0], kernel=3))
        self.base.append(ConvLayer(first_ch[0], first_ch[1], kernel=3))
        self.base.append(ConvLayer(first_ch[1], first_ch[2], kernel=3))
#         self.base.append(ConvLayer(first_ch[2], first_ch[3], kernel=3))

        skip_connection_channel_counts = []
        ch = first_ch[2]
        for i in range(blks):
            if downSamp[i] == 1:
                self.base.append(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))  # self.base.append(nn.AvgPool2d(kernel_size=2, stride=2))  # nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
#                 self.base.append(nn.Sequential(nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1),
#                                                nn.BatchNorm2d(ch),
#                                                nn.LeakyReLU(True)))

            blk = HarDBlock(ch, gr[i], grmul, n_layers[i])
            ch = blk.get_out_ch()
            skip_connection_channel_counts.append(ch)
            self.base.append(blk)

            self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]

#         self.conv_1 = ConvLayer(558, 256, kernel=3)
        self.conv_1 = ConvLayer(646, 256, kernel=3)
        self.conv_2 = ConvLayer(256, 128, kernel=3)
        self.semantic_output = nn.Conv2d(128, nclasses, 1)
        if self.aux:
#             self.aux_head1 = nn.Conv2d(78, nclasses, 1)
#             self.aux_head2 = nn.Conv2d(160, nclasses, 1)
#             self.aux_head3 = nn.Conv2d(224, nclasses, 1)

            self.aux_head1 = nn.Conv2d(130, nclasses, 1)
            self.aux_head2 = nn.Conv2d(130, nclasses, 1)
            self.aux_head3 = nn.Conv2d(128, nclasses, 1)


    def v2_transform(self):
        for i in range(len(self.base)):
            if isinstance(self.base[i], HarDBlock):
                blk = self.base[i]
                self.base[i] = HarDBlock_v2(blk.in_channels, blk.growth_rate, blk.grmul, blk.n_layers, list_out=True)
                self.base[i].transform(blk)
            elif isinstance(self.base[i], nn.Sequential):
                blk = self.base[i]
                sz = blk[0].weight.shape
                if sz[2] == 1:
                    self.base[i] = CatConv2d(sz[1], sz[0], (1, 1), relu=True)
                    self.base[i].weight[:, :, :, :] = blk[0].weight[:, :, :, :]
                    self.base[i].bias[:] = blk[0].bias[:]

        for i in range(self.n_blocks):
            blk = self.denseBlocksUp[i]
            self.denseBlocksUp[i] = HarDBlock_v2(blk.in_channels, blk.growth_rate, blk.grmul, blk.n_layers,
                                                 list_out=False)
            self.denseBlocksUp[i].transform(blk)

        for i in range(len(self.conv1x1_up)):
            blk = self.conv1x1_up[i]
            sz = blk[0].weight.shape
            if sz[2] == 1:
                self.conv1x1_up[i] = CatConv2d(sz[1], sz[0], (1, 1), relu=True)
                self.conv1x1_up[i].weight[:, :, :, :] = blk[0].weight[:, :, :, :]
                self.conv1x1_up[i].bias[:] = blk[0].bias[:]

    def forward(self, x):
        skip_connections = []
        size_in = x.size()

        for i in range(len(self.base)):
            x = self.base[i](x)
            if i in self.shortcut_layers:
                skip_connections.append(x)

        # for i in skip_connections:
        #     print(i.size())
        # print(x.size())
        # print("*"*50)


        res_1 = skip_connections.pop(0)            # x_0
        res_2 = skip_connections.pop(0)            # x_0

        res_3 = skip_connections.pop(0)            # x_2
        res_3 = F.interpolate(res_3, size=size_in[2:], mode='bilinear', align_corners=True)

        res_4 = skip_connections.pop(0)           # x_4
        res_4 = F.interpolate(res_4, size=size_in[2:], mode='bilinear', align_corners=True)

        res_5 = x         # x_8
        res_5 = F.interpolate(res_5, size=size_in[2:], mode='bilinear', align_corners=True)

        res = [res_1, res_2, res_3, res_4, res_5]

#         for i in res:
#             print(i.shape)

        out = torch.cat(res, dim=1)



        out = self.conv_1(out)
        out = self.conv_2(out)
        out = self.semantic_output(out)
        out = F.softmax(out, dim=1)

        if self.aux:
            res_3 = self.aux_head1(res_3)
            res_3 = F.softmax(res_3, dim=1)

            res_4 = self.aux_head2(res_4)
            res_4 = F.softmax(res_4, dim=1)

            res_5 = self.aux_head3(res_5)
            res_5 = F.softmax(res_5, dim=1)

        if self.aux:
            return [out, res_3, res_4, res_5]
        else:
            return out



# import time
# model = HarDNet()
# print(model)
# pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print("Number of parameters: ", pytorch_total_params / 1000000, "M")
# a = torch.randn([1, 5, 64, 512])
# temp = time.time()
# x = model(a)
# torch.cuda.synchronize()
# res = time.time() - temp
# print(res)
