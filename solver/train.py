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
from solver.loss import calc_loss

########################training##############################
def cls_eval(cls_preds, cls_labels):
    # 由于类别预测结果放在最后一维，argmax需要指定最后一维。
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())

class Accumulator:
    """
    在‘n’个变量上累加
    """

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def _getitem_(self, idx):
        return self.data[idx]


def trainnet(net,train_iter,args):
    trainer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_epochs = args.epochs  # 20
    if args.wandb_use == True:
        import wandb
        wandb.login(key='68b7dd90cbb4340172d092c05b0570186ee25a62')
        wandb.init(project="tinyssd", entity="dkhonker")
    for epoch in range(num_epochs):
        print('epoch: ', epoch)
        # 训练精确度的和，训练精确度的和中的示例数
        # 绝对误差的和，绝对误差的和中的示例数
        metric = Accumulator(4)
        net.train()
        for features, target in train_iter:
            trainer.zero_grad()
            X, Y = features.to(args.device), target.to(args.device)
            # 生成多尺度的锚框，为每个锚框预测类别和偏移量
            anchors, cls_preds, bbox_preds = net(X)
            # 为每个锚框标注类别和偏移量
            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
            # 根据类别和偏移量的预测和标注值计算损失函数
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                          bbox_masks)
            l.mean().backward()
            trainer.step()
            metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                       bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                       bbox_labels.numel())
        cls_err, bbox_mae = 1 - metric.data[0] / metric.data[1], metric.data[2] / metric.data[3]
        print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
        if args.wandb_use==True:
            wandb.log({
                "train/cls_loss": cls_err,
                "train/bbox_mae": bbox_mae,
            })

        # 保存模型参数
        if (epoch+1) % 10 == 0:
            torch.save(net.state_dict(), 'weights/net_' + str(epoch+1) + '.pkl')
    if args.wandb_use == True:
        wandb.finish(0)