from torchvision import models
from torchsummary import summary
from collections import OrderedDict
import torch.nn as nn
import torch


class Resnet50(nn.Module):
    def __init__(self, freeze=False):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([*list(models.resnet50(pretrained=True).named_children())[:-2]]))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 101)

        self.init_fc()
        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).reshape(x.shape[0], -1)

        x = self.fc(x)

        return x

    def init_fc(self):
        for m in self.fc.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)


class Resnet152(nn.Module):
    def __init__(self, freeze=False):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([*list(models.resnet152(pretrained=True).named_children())[:-2]]))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 101)

        self.init_fc()
        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).reshape(x.shape[0], -1)

        x = self.fc(x)

        return x

    def init_fc(self):
        for m in self.fc.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)


class Mobilenet_v2(nn.Module):
    def __init__(self, freeze=False):
        super(Mobilenet_v2, self).__init__()

        self.features = nn.Sequential(
            OrderedDict([*list(models.mobilenet_v2(pretrained=True).features.named_children())]))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 101),
        )
        self.init_fc()
        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).reshape(x.shape[0], -1)

        x = self.classifier(x)

        return x

    def init_fc(self):
        for m in self.classifier.modules():
            if hasattr(m, 'weight'):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')


class Densenet_121(nn.Module):
    def __init__(self, freeze=False):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([*list(models.densenet121(pretrained=True).features.named_children()),
                                                   ('relu', nn.ReLU(inplace=True))]),
                                      )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 101)
        self.init_fc()
        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def init_fc(self):
        for m in self.fc.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    m = Densenet_121()
    summary(m, (3, 224, 224), device='cpu')
