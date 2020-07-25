from torchvision import models
from torchsummary import summary
from collections import OrderedDict
import torch.nn as nn
import torch


class Resnet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([*list(models.resnet50(pretrained=True).named_children())[:-2]]))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 101)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    m = Resnet50().cuda()
    summary(m, (3, 224, 224))
