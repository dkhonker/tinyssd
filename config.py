import argparse
import json
import os
import random
import shutil
import sys

import numpy as np
import torch
#from munch import Munch
from torch.backends import cudnn

from utils.file import prepare_dirs, list_sub_folders
from utils.file import save_json
from utils.misc import get_datetime, str2bool, get_commit_hash, start_tensorboard


def setup_cfg(args):
    cudnn.benchmark = args.cudnn_benchmark
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    if args.mode == 'train' and torch.cuda.device_count() > 1:
        print(f"We will train on {torch.cuda.device_count()} GPUs.")
        args.multi_gpu = True
    else:
        args.multi_gpu = False

    if args.mode == 'train':
        if args.exp_id is None:
            args.exp_id = get_datetime()
            # Tip: you can construct the exp_id automatically here by use the args.
    else:
        if args.exp_id is None:
            args.exp_id = input("Please input exp_id: ")
        if not os.path.exists(os.path.join(args.exp_dir, args.exp_id)):
            all_existed_ids = os.listdir(args.exp_dir)
            for existed_id in all_existed_ids:
                if existed_id.startswith(args.exp_id + "-"):
                    args.exp_id = existed_id
                    print(f"Warning: exp_id is reset to {existed_id}.")
                    break

    if args.debug:
        print("Warning: running in debug mode, some settings will be override.")
        args.exp_id = "debug"
        args.sample_every = 10
        args.eval_every = 20
        args.save_every = 20
        args.end_iter = args.start_iter + 60
    if os.name == 'nt' and args.num_workers != 0:
        print("Warning: reset num_workers = 0, because running on a Windows system.")
        args.num_workers = 0

    args.log_dir = os.path.join(args.exp_dir, args.exp_id, "logs")
    args.sample_dir = os.path.join(args.exp_dir, args.exp_id, "samples")
    args.model_dir = os.path.join(args.exp_dir, args.exp_id, "models")
    args.eval_dir = os.path.join(args.exp_dir, args.exp_id, "eval")
    prepare_dirs([args.log_dir, args.sample_dir, args.model_dir, args.eval_dir])
    args.record_file = os.path.join(args.exp_dir, args.exp_id, "records.txt")
    args.loss_file = os.path.join(args.exp_dir, args.exp_id, "losses.csv")

    if os.path.exists(f'./scripts/{args.exp_id}.sh'):
        shutil.copyfile(f'./scripts/{args.exp_id}.sh', os.path.join(args.exp_dir, args.exp_id, f'{args.exp_id}.sh'))

    if args.mode == 'train' and args.start_tensorboard:
        start_tensorboard(os.path.join(args.exp_dir, args.exp_id), 'logs')

    args.domains = list_sub_folders(args.train_path, full_path=False)
    args.num_domains = len(args.domains)



def print_cfg(cfg):
    print(type(cfg))
    print(json.dumps(cfg.__dict__, indent=4))


def parse_args():
    parser = argparse.ArgumentParser()

    # Meta arguments.
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # Model related arguments.
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--init_weights', type=str, default='he', choices=['he', 'default'])

    # Training related arguments
    parser.add_argument('--batch_size', type=int, default=64)
    #parser.add_argument('--start_iter', type=int, default=0)
    #parser.add_argument('--end_iter', type=int, default=200000)
    parser.add_argument('--epochs', type=int, default=3)


    # Optimizing related arguments.
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for generator.")
    parser.add_argument('--weight_decay', type=float, default=1e-4)


    # Step related arguments.
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--wandb_use', type=bool, default=False)

    # Log related arguments.
    #parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    #parser.add_argument('--start_tensorboard', type=str2bool, default=False)
    #parser.add_argument('--save_loss', type=str2bool, default=True)

    # Others
    parser.add_argument('--seed', type=int, default=3407, help='Seed for random number generator.')
    parser.add_argument('--pretrain', type=str, default=None,
                        help='The name list of the pretrained models that you used.')

    return parser.parse_args()
