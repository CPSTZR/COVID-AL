import torch.nn as nn
from modules.FeatureSelection import FeatureSelectionLayer

from torchvision import models


class RACNN(nn.Module):
    def __init__(self, num_classes):
        super(RACNN, self).__init__()

        VGG19 = models.vgg19(pretrained=True)

        self.V19_e1 = VGG19.features
        self.V19_e2 = VGG19.features

        self.GAP1 = nn.AdaptiveAvgPool2d(output_size=1)
        self.GAP2 = nn.AdaptiveAvgPool2d(output_size=1)

        self.atten_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.apn = nn.Sequential(
            nn.Linear(512 * 14 * 14, 1024),
            nn.Tanh(),
            nn.Linear(1024, 3),
            nn.Sigmoid(),
        )
        self.feature_selection = FeatureSelectionLayer()

        self.fn1 = nn.Linear(512, num_classes)
        self.fn2 = nn.Linear(512, num_classes)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        conv5_4 = self.V19_e1[:-1](x)
        pool5 = self.GAP1(conv5_4)
        fe_r = self.apn(self.atten_pool(conv5_4).view(-1, 512 * 14 * 14))
        scaledA_x = self.feature_selection(x,  fe_r * 448)

        conv5_4_A = self.V19_e2[:-1](scaledA_x)
        pool5_A = self.GAP2(conv5_4_A)

        pool5 = pool5.view(-1, 512)
        pool5_A = pool5_A.view(-1, 512)

        logits1 = self.fn1(pool5)
        logits2 = self.fn2(pool5_A)

        s_logis1 = self.softmax(logits1)
        s_logis2 = self.softmax(logits2)
        return [logits1, logits2], [s_logis1, s_logis2], [conv5_4, conv5_4_A], atten1, scaledA_x