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

from data.dataset import Dataset


def load_data(batch_size):
    """加载检测数据集"""
    train_iter = torch.utils.data.DataLoader(Dataset(is_train=True),
                                             batch_size, shuffle=True)
    # val_iter = torch.utils.data.DataLoader(Dataset(is_train=False),
    #                                       batch_size)
    return train_iter  # , val_iter