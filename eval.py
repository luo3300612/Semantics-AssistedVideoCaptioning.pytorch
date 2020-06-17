import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam
# from optim import Adam
from dataloader import DataManager, collate_fn
from torch.utils.data import DataLoader
from functools import partial
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from tensorboardX import SummaryWriter
import os
from models import init_model
from utils import NLLLossWithLength
from torch.optim.lr_scheduler import ExponentialLR
from opts import get_eval_args
from train import evaluate

if __name__ == '__main__':
    # get configs
    args = get_eval_args()
    print(args)

    # load data
    dm = DataManager(args)

    # prepare model
    model = init_model(args, dm)
    model = model.cuda()
    model.load_state_dict(torch.load(args.model_path))

    # split data
    _, _, test_data = dm.split()
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.num_workers,
                             collate_fn=partial(collate_fn, split='test', padding_idx=dm.word2idx['<pad>']))
    print('Start Video-Captioning Evaluation')

    test_score = evaluate(0, model, test_loader, dm, maxlen=args.max_sent_len, split='Test', verbose=True)
    print('Test score:')
    print(test_score)
    print('Done')
