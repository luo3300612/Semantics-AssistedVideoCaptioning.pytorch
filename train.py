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
from opts import get_args,save_args

def train_epoch(epoch, model, dataloader, optimizer, loss_fn):
    model.train()
    running_loss = 0.0
    with tqdm(desc='Epoch {} - train'.format(epoch), unit='it', total=len(dataloader)) as pbar:
        for it, (feats, tags, captions) in enumerate(dataloader):
            feats = feats.cuda()
            tags = tags.cuda()
            captions = captions.cuda()

            optimizer.zero_grad()
            out = model(feats, tags, captions)
            loss = loss_fn(out, captions)
            # loss = loss_fn(out.view(-1, out.shape[-1]), captions.view(-1))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
    loss = running_loss / len(dataloader)
    return loss


def evaluate(epoch, model, dataloader, dm, maxlen, split, verbose=False):
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Epoch {} - {}'.format(epoch, split), unit='it', total=len(dataloader)) as pbar:
        for it, (feats, tags, captions) in enumerate(dataloader):
            feats = feats.cuda()
            tags = tags.cuda()
            with torch.no_grad():
                out = model.sample(feats, tags, maxlen=maxlen)
            decoded_sentences = dm.decode(out)
            for i in range(len(decoded_sentences)):
                gen['{}_{}'.format(it, i)] = [decoded_sentences[i]]
                gts['{}_{}'.format(it, i)] = captions[i]
            pbar.update()

    if verbose:
        for key in gts.keys():
            print('gen:', gen[key][0])

    return score(gts, gen)


def score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    # print('ref')
    # print(ref)
    # print('hypo')
    # print(hypo)
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores


if __name__ == '__main__':
    # get configs
    args = get_args()
    print(args)

    writer = SummaryWriter(log_dir=os.path.join(args.savedir, args.exp_name))
    save_args(args,os.path.join(args.savedir,args.exp_name))

    # load data
    dm = DataManager(args)

    # prepare model
    model = init_model(args, dm)
    model = model.cuda()

    # prepare training
    optimizer = Adam(lr=args.lr, params=model.parameters())
    schedule = ExponentialLR(optimizer, args.lr_decay)
    loss_fn = NLLLossWithLength(ignore_index=dm.word2idx['<pad>'])
    # loss_fn = nn.NLLLoss(ignore_index=dm.word2idx['<pad>'])

    # split data
    train_data, val_data, test_data = dm.split()
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              collate_fn=partial(collate_fn, split='train', padding_idx=dm.word2idx['<pad>']))
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.num_workers,
                            collate_fn=partial(collate_fn, split='val', padding_idx=dm.word2idx['<pad>']))
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.num_workers,
                             collate_fn=partial(collate_fn, split='test', padding_idx=dm.word2idx['<pad>']))
    print('Start Video-Captioning Training')

    best_cider = 0
    best_epoch = -1
    best_score = None

    max_epoch = args.n_epoch
    for i in range(max_epoch):
        model.schedule_sample_prob = args.schedule_sample_prob + i * args.schedule_sample_ratio
        train_loss = train_epoch(i, model, train_loader, optimizer, loss_fn)
        writer.add_scalar('train/loss', train_loss, i)
        writer.add_scalar('train/schedule_sample_prob', model.schedule_sample_prob, i)
        writer.add_scalar('train/lr', optimizer.state_dict()['param_groups'][0]['lr'], i)
        print('Train loss:', train_loss)
        if (i + 1) % args.lr_decay_every == 0:
            schedule.step()

        # eval on val
        val_score = evaluate(i, model, val_loader, dm, maxlen=args.max_sent_len, split='Val')
        writer.add_scalar('val/Bleu1', val_score['Bleu_1'], i)
        writer.add_scalar('val/Bleu2', val_score['Bleu_2'], i)
        writer.add_scalar('val/Bleu3', val_score['Bleu_3'], i)
        writer.add_scalar('val/Bleu4', val_score['Bleu_4'], i)
        writer.add_scalar('val/RougeL', val_score['ROUGE_L'], i)
        writer.add_scalar('val/METEOR', val_score['METEOR'], i)
        writer.add_scalar('val/CIDEr', val_score['CIDEr'], i)
        print('Val score:')
        print(val_score)

        # eval on test
        test_score = evaluate(i, model, test_loader, dm, maxlen=args.max_sent_len, split='Test', verbose=True)
        writer.add_scalar('test/Bleu1', test_score['Bleu_1'], i)
        writer.add_scalar('test/Bleu2', test_score['Bleu_2'], i)
        writer.add_scalar('test/Bleu3', test_score['Bleu_3'], i)
        writer.add_scalar('test/Bleu4', test_score['Bleu_4'], i)
        writer.add_scalar('test/RougeL', test_score['ROUGE_L'], i)
        writer.add_scalar('test/METEOR', test_score['METEOR'], i)
        writer.add_scalar('test/CIDEr', test_score['CIDEr'], i)
        print('Test score:')
        print(test_score)

        if val_score['CIDEr'] > best_cider:
            best_cider = val_score['CIDEr']
            best_score = test_score
            best_epoch = i
            torch.save(model.state_dict(), os.path.join(args.savedir, args.exp_name, 'best.pth'))

        torch.save(model.state_dict(), os.path.join(args.savedir, args.exp_name, 'last.pth'))

    writer.add_scalar('best/Bleu1', best_score['Bleu_1'], 1)
    writer.add_scalar('best/Bleu2', best_score['Bleu_2'], 1)
    writer.add_scalar('best/Bleu3', best_score['Bleu_3'], 1)
    writer.add_scalar('best/Bleu4', best_score['Bleu_4'], 1)
    writer.add_scalar('best/RougeL', best_score['ROUGE_L'], 1)
    writer.add_scalar('best/METEOR', best_score['METEOR'], 1)
    writer.add_scalar('best/CIDEr', best_score['CIDEr'], 1)
    writer.close()
    print('Best epoch', best_epoch)
    print('Best cider', best_cider)
    print('Best score', best_score)
    print('Done')
