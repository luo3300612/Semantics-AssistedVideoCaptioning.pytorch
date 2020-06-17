import argparse
import yaml
import os

data_keys = ['dataset', 'corpus', 'reseco', 'tag', 'ref', 'val_start_idx', 'val_end_idx',
             'test_start_idx', 'test_end_idx', 'num_workers']

model_keys = ['model', 'embedding_dim', 'hidden_dim']

checkpoint_keys = ['exp_name', 'savedir']


def get_args():
    parser = argparse.ArgumentParser(description='Video Captioning Arguments', argument_default=argparse.SUPPRESS)

    # data
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--corpus', type=str)
    parser.add_argument('--reseco', type=str)
    parser.add_argument('--tag', type=str)
    parser.add_argument('--ref', type=str)
    parser.add_argument('--num_workers', type=str)

    # model
    parser.add_argument('--model', type=str)
    parser.add_argument('--embedding_dim', type=int)
    parser.add_argument('--hidden_dim', type=int)

    # learning
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_decay', type=float)
    parser.add_argument('--lr_decay_every', type=int, help='epoch')
    parser.add_argument('--n_epoch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--max_sent_len', type=int)

    # checkpoints
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--savedir', type=str)

    # sample
    parser.add_argument('--schedule_sample_method', type=str)
    parser.add_argument('--schedule_sample_prob', type=float)
    parser.add_argument('--schedule_sample_ratio', type=float)

    # config
    parser.add_argument('--cfg', type=str, default=None)

    args = parser.parse_args()

    # load config in yaml
    if args.cfg is not None:
        args = load_args(args, args.cfg)

    # check args
    assert args.schedule_sample_method in ['greedy', 'multinomial']

    return args


def get_eval_args():
    parser = argparse.ArgumentParser(description='Video Captioning Arguments', argument_default=argparse.SUPPRESS)

    # data
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--corpus', type=str)
    parser.add_argument('--reseco', type=str)
    parser.add_argument('--tag', type=str)
    parser.add_argument('--ref', type=str)
    parser.add_argument('--num_workers', type=str)

    # model
    parser.add_argument('--model', type=str)
    parser.add_argument('--embedding_dim', type=int)
    parser.add_argument('--hidden_dim', type=int)

    # learning
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_decay', type=float,help='lr decay ratio')
    parser.add_argument('--lr_decay_every', type=int, help='epoch')
    parser.add_argument('--n_epoch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--max_sent_len', type=int)

    # checkpoints
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--savedir', type=str)

    # sample
    parser.add_argument('--schedule_sample_method', type=str)
    parser.add_argument('--schedule_sample_prob', type=float)
    parser.add_argument('--schedule_sample_ratio', type=float)

    args = parser.parse_args()

    # load config in yaml
    args = load_args(args, os.path.join(args.savedir, args.exp_name, 'configs.yaml'))

    if not hasattr(args, 'model_path'):
        setattr(args, 'model_path', os.path.join(args.savedir, args.exp_name, 'best.pth'))

    return args


def save_args(args, path):
    d = vars(args)
    with open(os.path.join(path, 'configs.yaml'), 'w') as f:
        yaml.dump(d, f)


def load_args(args, path):
    configs = yaml.safe_load(open(path, 'r'))
    # needed_args = data_keys + model_keys + checkpoint_keys
    for k, v in configs.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    return args


if __name__ == '__main__':
    args = get_args()
    print('configs')
    print(args)
    save_args(args, './')
