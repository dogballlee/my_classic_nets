import torch.nn as nn
from torch.nn import Conv2d
from torch.nn import functional as f
from collections import OrderedDict


def num_flat_features(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv_unit = nn.Sequential(OrderedDict([("conv1", Conv2d(1, 6,
                                                                     kernel_size=(5, 5),
                                                                     stride=(1, 1),
                                                                     padding=(2, 2),
                                                                     bias=True)),
                                                    ("conv2", Conv2d(6, 16,
                                                                     kernel_size=(5, 5),
                                                                     stride=(1, 1))),
                                                    ]))

        self.fc_unit = nn.Sequential(OrderedDict([("fc1", nn.Linear(16*5*5, 120)),
                                                  ("fc2", nn.Linear(120, 84)),
                                                  ("fc3", nn.Linear(84, 10)),
                                                  ]))

    def forward(self, x):
        # 第一次卷积+池化
        x = f.avg_pool2d(f.leaky_relu(self.conv_unit.conv1(x)), (2, 2))
        # 第二次卷积+池化
        x = f.avg_pool2d(f.leaky_relu(self.conv_unit.conv2(x)), (2, 2))
        x = x.view(-1, num_flat_features(x))
        # print('size', x.size())
        # 第一次全连接
        x = f.relu(self.fc_unit.fc1(x))
        # 第二次全连接
        x = f.relu(self.fc_unit.fc2(x))
        x = self.fc_unit.fc3(x)
        return x
