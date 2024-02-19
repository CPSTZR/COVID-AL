import torch
import torch.nn as nn
from modules.FeatureSelection import FeatureSelectionLayer
from modules.ResNet import AM_resnet18
from torchvision import models


class CovidNet(nn.Module):
    def __init__(self, num_classes):
        super(CovidNet, self).__init__()

        VGG19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.V19_e1 = VGG19.features[:-1]
        self.atten_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.apn = nn.Sequential(
            nn.Linear(512 * 14 * 14, 1024),
            nn.Tanh(),
            nn.Linear(1024, 3),
            nn.Sigmoid(),
        )
        self.feature_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.feature_selection = FeatureSelectionLayer()
        AM_RESNET18 = AM_resnet18(pretrained=False)
        AM_RESNET18.fc = nn.Linear(512, 3, bias=True)
        self.classifier = AM_RESNET18

    def forward(self, x):
        conv5_4 = self.V19_e1 =(x)
        conv5_4_ap = self.atten_pool(conv5_4)
        fe_r = self.apn(conv5_4_ap.view(-1, 512 * 14 * 14))
        scaledA_x = self.feature_selection(x, fe_r * 448)
        c = self.classifier(scaledA_x)
        return c



