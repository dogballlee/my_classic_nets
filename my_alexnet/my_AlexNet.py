import torch.nn as nn
from torch.nn import Conv2d
import torch.nn.functional as f
from collections import OrderedDict


def num_flat_features(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class MyAlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(OrderedDict([('conv1', Conv2d(3, 96, (11, 11), (4, 4), (2, 2))),
                                                 ('max_pool', nn.MaxPool2d((3, 3), 2, 0)),
                                                 ('conv2', Conv2d(96, 256, (5, 5), (1, 1), (2, 2))),
                                                 ('conv3', Conv2d(256, 384, (3, 3), (1, 1), (1, 1))),
                                                 ('conv4', Conv2d(384, 384, (3, 3), (1, 1), (1, 1))),
                                                 ('conv5', Conv2d(384, 256, (3, 3), (1, 1), (1, 1))),
                                                 ('drop_out', nn.Dropout(0.5)),
                                                 ('fc1', nn.Linear(9216, 4096)),
                                                 ('fc2', nn.Linear(4096, 4096)),
                                                 ('fc3', nn.Linear(4096, 10))]))

    def forward(self, x):
        x = self.layers.max_pool(f.relu(self.layers.conv1(x)))
        x = self.layers.max_pool(f.relu(self.layers.conv2(x)))
        x = f.relu(self.layers.conv3(x))
        x = f.relu(self.layers.conv4(x))
        x = self.layers.max_pool(f.relu(self.layers.conv5(x)))
        x = x.view(-1, num_flat_features(x))
        x = self.layers.drop_out(f.relu(self.layers.fc1(x)))
        x = self.layers.drop_out(f.relu(self.layers.fc2(x)))
        x = self.layers.fc3(x)
        return x
