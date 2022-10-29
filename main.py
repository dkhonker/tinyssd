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
import random
import numpy as np

from utils.utils import *

from config import  print_cfg,parse_args
from solver.train import trainnet
from solver.test import testnet

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def main(args):
    from models.model import TinySSD
    net = TinySSD(num_classes=1,args=args)
    net = net.to(args.device)

    if args.mode == 'train':
        if args.pretrain:
            net.load_state_dict(torch.load(args.pretrain, map_location=args.device))
        import time
        time_start = time.time()  # 记录开始时间

        print("-"*7+"正在训练"+"-"*7)
        from data.loader import load_data
        train_iter = load_data(args.batch_size)
        trainnet(net, train_iter, args)
        time_end = time.time()  # 记录结束时间
        time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
        print(time_sum/60,"分钟")
    if args.mode == 'test':
        testnet(net, args)

if __name__ == '__main__':
    import torch
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    cfg = parse_args()
    print_cfg(cfg)
    seed_everything(cfg.seed)
    main(cfg)