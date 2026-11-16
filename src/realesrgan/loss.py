from collections import OrderedDict
import torch
from torch import nn
from torchvision.models import VGG19_Weights, vgg19
import config

VGG19_LAYERS = [
    "conv1_1", "relu1_1", "conv1_2", "relu1_2", "pool1",
    "conv2_1", "relu2_1", "conv2_2", "relu2_2", "pool2",
    "conv3_1", "relu3_1", "conv3_2", "relu3_2", "conv3_3", "relu3_3", "conv3_4", "relu3_4", "pool3",
    "conv4_1", "relu4_1", "conv4_2", "relu4_2", "conv4_3", "relu4_3", "conv4_4", "relu4_4", "pool4",
    "conv5_1", "relu5_1", "conv5_2", "relu5_2", "conv5_3", "relu5_3", "conv5_4", "relu5_4", "pool5"
]


class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_name_list, use_input_norm=True, range_norm=False):
        super().__init__()
        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        max_idx = max(VGG19_LAYERS.index(v) for v in layer_name_list)
        vgg_net = vgg19(weights=VGG19_Weights.DEFAULT)
        features = vgg_net.features[:max_idx + 1]

        modified_net = OrderedDict()
        for k, v in zip(VGG19_LAYERS, features):
            modified_net[k] = v

        self.vgg_net = nn.Sequential(modified_net).eval().to(config.DEVICE)
        for param in self.parameters():
            param.requires_grad = False

        if self.use_input_norm:
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=config.DEVICE).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=config.DEVICE).view(1, 3, 1, 1))

    def forward(self, x):
        if self.range_norm:
            x = (x + 1) / 2
        if self.use_input_norm:
            x = (x - self.mean) / self.std

        output = {}
        for key, layer in self.vgg_net._modules.items():
            x = layer(x)
            if key in self.layer_name_list:
                output[key] = x.clone()
        return output


class PerceptualLoss(nn.Module):
    def __init__(self, layer_weights, perceptual_weight=1.0):
        super().__init__()
        self.perceptual_weight = perceptual_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(layer_name_list=list(layer_weights.keys()))
        self.criterion = nn.L1Loss()

    def forward(self, x, gt):
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        percep_loss = 0.0
        for k in x_features.keys():
            percep_loss = percep_loss + self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
        return percep_loss * self.perceptual_weight


class L1Loss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        return self.loss_weight * nn.functional.l1_loss(pred, target, reduction="mean")
