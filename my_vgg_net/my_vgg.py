import torch.nn as nn
# import torch.nn.functional as f
# from collections import OrderedDict


# 实现一个VGG块
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for i in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), (1, 1)))     # 保持高宽的卷积方式
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d((2, 2), (2, 2)))     # 为了使每个块后的分辨率减半
    return nn.Sequential(*layers)


conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))


def vgg(ca):
    conv_blks = []
    in_channels = 1
    out_channels = 0        # 此行不写的话，第30行会报警告，无大碍但是会让人很不爽
    # 卷积层
    for (num_convs, out_channels) in ca:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(*conv_blks, nn.Flatten(),
                         nn.Linear(out_channels * 7 * 7, 4096),
                         nn.ReLU(),
                         nn.Dropout(0.5),
                         nn.Linear(4096, 4096),
                         nn.ReLU(),
                         nn.Dropout(0.5),
                         nn.Linear(4096, 10))


net = vgg(conv_arch)
