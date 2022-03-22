import torch as th
import numpy as np
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt


class my_res18_basicblock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1))
        self.bn0 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv0(x)
        output = F.relu(self.bn0(output))
        output = self.conv1(output)
        output = self.bn1(output)
        return F.relu(x + output)


class my_res18_downblock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride[0], padding=(1, 1))
        self.bn0 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=stride[1], padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride[0], padding=(0, 0)),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv0(x)
        out = F.relu(self.bn0(output))

        out = self.conv1(out)
        out = self.bn0(out)
        return F.relu(extra_x + out)


class my_res18(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.bn0 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer0 = nn.Sequential(my_res18_basicblock(64, 64, 1),
                                    my_res18_basicblock(64, 64, 1))

        self.layer1 = nn.Sequential(my_res18_downblock(64, 128, [2, 1]),
                                    my_res18_basicblock(128, 128, 1))

        self.layer2 = nn.Sequential(my_res18_downblock(128, 256, [2, 1]),
                                    my_res18_basicblock(256, 256, 1))

        self.layer3 = nn.Sequential(my_res18_downblock(256, 512, [2, 1]),
                                    my_res18_basicblock(512, 512, 1))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out = self.conv0(x)
        out = self.layer0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
