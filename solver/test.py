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

#########################test##############################








def predict(X,net,device):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

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

def testnet(net,args):
    #net.load_state_dict(torch.load(args.pretrain, map_location=args.device))
    net.load_state_dict(torch.load(args.pretrain, map_location=args.device))
    files = glob.glob('dataset/test/*.jpg')
    i = 1
    for name in files:
        plt.subplot(1, 2, i)
        i += 1
        X = torchvision.io.read_image(name).unsqueeze(0).float()
        img = X.squeeze(0).permute(1, 2, 0).long()

        output = predict(X,net,args.device)
        if args.backbone=="vgg":
            display(img, output.cpu(), threshold=0.42)
        elif args.backbone == "resnet":
            display(img, output.cpu(), threshold=0.60)
        else:
            display(img, output.cpu(), threshold=0.45)
    if args.CBAM ==True:
        plt.savefig("demo/"+args.backbone+"+CBAM.jpg")
    else:
        plt.savefig("demo/" + args.backbone + ".jpg")
    #plt.show()
        # break