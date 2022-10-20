# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 16:50:15 2022

@author: nkliu
"""

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
from utils.utils import *





batch_size = 32
from data.loader import load_data
train_iter = load_data(batch_size)


from models.model import TinySSD
net = TinySSD(num_classes=1)
net = net.to('cuda')


from solver.solver import trainnet
trainnet(net,train_iter)

from solver.test import predict
def show_bboxes(axes, bboxes, labels=None, colors=None):
    """显示所有边界框"""

    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))




def display(img, output, threshold):
    fig = plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

files = glob.glob('dataset/test/*.jpg')
i = 1
for name in files:
    plt.subplot(1, 2, i)
    i += 1
    X = torchvision.io.read_image(name).unsqueeze(0).float()
    img = X.squeeze(0).permute(1, 2, 0).long()

    output = predict(X,net)
    display(img, output.cpu(), threshold=0.5)
    # break