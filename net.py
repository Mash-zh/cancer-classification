import torch
import torchvision.models as models
import torch.nn as nn

class CustomNet(nn.Module):
    def __init__(self, net_name, pretrained=True, num_classes=2):
        super(CustomNet, self).__init__()
        self.net = net_name
        self.num_classes = num_classes
        self.pretrained = pretrained

    def __net__(self):
        if self.net == 'resnet18':
            net = models.resnet18(pretrained=self.pretrained)
            net.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
            net.fc = nn.Linear(net.fc.in_features, self.num_classes)
        elif self.net == 'resnet34':
            net = models.resnet34(pretrained=self.pretrained)
            net.fc = nn.Linear(net.fc.in_features, self.num_classes)
        elif self.net == 'vgg16':
            net = models.vgg16(pretrained=self.pretrained)
            net.classifier[6] = nn.Linear(net.classifier[6].in_features, self.num_classes)
        elif self.net == 'efficientnet_v2_m':
            net = models.efficientnet_v2_m(pretrained=self.pretrained)
            net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, self.num_classes)
        elif self.net == 'inception3':
            net = models.inception_v3(pretrained=self.pretrained)
            net.fc = nn.Linear(net.fc.in_features, self.num_classes)
        else:
            print("error net name!")
        return net
