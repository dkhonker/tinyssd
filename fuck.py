import torch
import torchvision
import os
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import glob

def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)

def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i + 1]))
    return nn.Sequential(*blk)


def conv_block(in_ch, out_ch, pool=False):
    '''Represents single convolution block'''
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
              nn.BatchNorm2d(out_ch),
              nn.ReLU()]
    if pool:
        layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self):
        super().__init__()
        # in 32*32*3
        self.conv1 = conv_block(3, 8)  # 32*32*64
        self.conv2 = conv_block(8, 16, True)  # 16*16*128

        self.res1 = nn.Sequential(conv_block(16, 16),
                                  conv_block(16, 16))
        self.conv3 = conv_block(16, 32, True)  # 8*8*256
        self.conv4 = conv_block(32, 64, True)  # 4*4*512
        self.res2 = nn.Sequential(conv_block(64, 64),
                                  conv_block(64, 64))

    def forward(self, x):
        print("jhas")
        out = self.conv1(x)
        print(out.shape)
        out = self.conv2(out)
        print(out.shape)
        out = self.res1(out) + out
        print(out.shape)
        out = self.conv3(out)
        print(out.shape)
        out = self.conv4(out)
        print(out.shape)
        out = self.res2(out) + out
        print(out.shape)
        return out



tmp = torch.randn(8,3,256,256)
model1=ResNet9()
model2=base_net()
print(model2(tmp).shape)
print(model1(tmp).shape)
