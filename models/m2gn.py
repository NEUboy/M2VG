import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base import BaseNetwork
from .xception import Xception
from .efficientnet import EfficientNet
from timm.models.layers import DropPath
from .modules.head import Classifier2D, Localizer
from .modules.Freq import FreqAttention
from .graph_modules.basic import act_layer
from .graph_modules.graph_vertex import DyGraphConv2d
from .graph_modules.pos_embed import get_2d_relative_pos_embed

from utils.registries import MODEL_REGISTRY

class CMA_Block(nn.Module):
    def __init__(self, rgb_channel, freq_channel, hidden_channel, out_channel):
        super(CMA_Block, self).__init__()

        self.conv1 = nn.Conv2d(
            rgb_channel, hidden_channel, kernel_size=1, stride=1, padding=0
        )
        self.conv2 = nn.Conv2d(
            freq_channel, hidden_channel, kernel_size=1, stride=1, padding=0
        )
        self.conv3 = nn.Conv2d(
            freq_channel, hidden_channel, kernel_size=1, stride=1, padding=0
        )

        self.scale = hidden_channel ** -0.5

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                hidden_channel, out_channel, kernel_size=1, stride=1, padding=0
            ),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, rgb, freq):
        _, _, h, w = rgb.size()

        q = self.conv1(rgb)
        k = self.conv2(freq)
        v = self.conv3(freq)

        q = q.view(q.size(0), q.size(1), q.size(2) * q.size(3)).transpose(
            -2, -1
        )
        k = k.view(k.size(0), k.size(1), k.size(2) * k.size(3))

        attn = torch.matmul(q, k) * self.scale
        m = attn.softmax(dim=-1)

        v = v.view(v.size(0), v.size(1), v.size(2) * v.size(3)).transpose(
            -2, -1
        )
        z = torch.matmul(m, v)
        z = z.view(z.size(0), h, w, -1)
        z = z.permute(0, 3, 1, 2).contiguous()

        output = rgb + self.conv4(z)

        return output

class Grapher(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=True):
        super(Grapher, self).__init__()
        self.n = n
        self.r = r
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels)
        )
        self.graph_conv = DyGraphConv2d(in_channels, in_channels *2, kernel_size, dilation, conv,
                                        act, norm, bias, stochastic, epsilon, r)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels *2, in_channels, kernel_size = 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.relative_pos = None
        if relative_pos:
            # print('using relative_pos')
            relative_pos_tensor = torch.from_numpy(
                np.float32(get_2d_relative_pos_embed(in_channels,int(n ** 0.5)))
            ).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(
                relative_pos_tensor, size=(n, n // (r * r)), mode='bicubic', align_corners=False
            )
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    def forward(self, x):
        tmp = x
        x = self.fc1(x)
        B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)
        x = self.graph_conv(x, relative_pos)
        x = self.fc2(x)
        x = self.drop_path(x) + tmp
        return x

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x  # .reshape(B, C, N, 1)

@MODEL_REGISTRY.register()
class M2GN(BaseNetwork):
    def __init__(self, model_cfg):
        super(M2GN,self).__init__()
        backbone = model_cfg["BACKBONE"]
        texture_layer = model_cfg["TEXTURE_LAYER"]
        feature_layer = model_cfg["FEATURE_LAYER"]
        graph_deepth = model_cfg["GRAPH_DEPTH"]
        knn = model_cfg["GRAPH_NUM"]
        drop_ratio = model_cfg["DROP_RATIO"]
        num_classes = model_cfg["NUM_CLASSES"]
        img_size = model_cfg["IMG_SIZE"]

        if "xception" in backbone:
            self.model = Xception(num_classes)
        elif backbone.split("-")[0] == "efficientnet":
            self.model = EfficientNet({'NAME': backbone, 'PRETRAINED': True})

        with torch.no_grad():
            input = {"img": torch.zeros(1, 3, img_size, img_size)}
            layers = self.model(input)

        texture_dim = layers[texture_layer].shape[1]
        feature_dim = layers[feature_layer].shape[1]

        self.pos_embed = nn.Parameter(torch.zeros(1, texture_dim, img_size//4, img_size//4))

        self.layers = nn.ModuleList([])
        num_knn = [int(x.item()) for x in torch.linspace(knn, 2 * knn, graph_deepth)]
        dpr = [x.item() for x in torch.linspace(0, drop_ratio, graph_deepth)]

        for i in range(graph_deepth):
            self.layers.append(
                nn.ModuleList([
                    # FreqAttention(in_channel=texture_dim, in_size=img_size // 4,  scale=4),
                    FreqAttention(in_channel=texture_dim, in_size= img_size//4, scale = 4 * int(math.pow(2, i))),
                    CMA_Block(rgb_channel=texture_dim, freq_channel = texture_dim,
                              hidden_channel=texture_dim, out_channel=texture_dim),
                    Grapher(texture_dim, num_knn[i], 1, "mr", "gelu", "batch",
                            True, True, 0.2, 1, drop_path=dpr[i]),
                    FFN(texture_dim, texture_dim * 4, act="gelu")
                ])
            )

        self.classifier = Classifier2D(
            feature_dim, num_classes, drop_ratio, "sigmoid"
        )

        # for param in self.model.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        x = x["img"]
        # x = x.repeat(10, 1, 1, 1)
        B = x.size(0)

        layers = {}
        rgb = self.model.extract_textures(x, layers)

        rgb += self.pos_embed

        for freq, cma, grapher, ffn in self.layers:
            freq_f = freq(rgb)
            # freq_f8 = freq_8(rgb)
            # rgb = cma(rgb, torch.cat([freq_f4, freq_f8], dim= 1))
            rgb = cma(rgb, freq_f)
            rgb = grapher(rgb)
            rgb = ffn(rgb)

        features = self.model.extract_features(rgb, layers)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(B, features.size(1))
        logits = self.classifier(features)

        output = {"logits": logits, "features": features, "map": rgb}
        return output

# ACC 82.73 AUC 72.8
# ACC 82.60 AUC 73.9
# Acc 82.685 Auc 74.3
# ACC 90.855 AUC 95.2
# 387.79 M FLOPs 26.57 M Param
