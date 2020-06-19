from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pickle
import torch


def collate_fn(batch, split, padding_idx):
    if split == 'train':
        max_len = max([len(caption) for _, _, caption in batch])
        batched_captions = np.ones((len(batch), max_len), dtype='int') * padding_idx
        for i, (_, _, caption) in enumerate(batch):
            batched_captions[i, :len(caption)] = caption
        batched_captions = torch.from_numpy(batched_captions).long()
    else:
        batched_captions = [caption for _, _, caption in batch]

    feats = [feat for feat, _, _ in batch]
    tags = [tag for _, tag, _ in batch]
    batched_feats = torch.from_numpy(np.stack(feats))
    batched_tags = torch.from_numpy(np.stack(tags))

    return batched_feats, batched_tags, batched_captions


class DataManager:
    def __init__(self, args):
        self.features = np.load(args.reseco)
        self.tags = np.load(args.tag)
        self.corpus = pickle.load(open(args.corpus,'rb'))
        self.refs = pickle.load(open(args.ref,'rb'))
        self.max_sent_len = args.max_sent_len

        self.val_start_idx = args.val_start_idx
        self.val_end_idx = args.val_end_idx
        self.test_start_idx = args.test_start_idx
        self.test_end_idx = args.test_end_idx

        self.idx2word = self.corpus[4]
        self.idx2word[len(self.idx2word)] = '<bos>'
        self.idx2word[len(self.idx2word)] = '<pad>'
        self.word2idx = {value: key for key, value in self.idx2word.items()}

    def get_train(self):
        train_data = self.corpus[0]
        train_feats = self.features
        train_tags = self.tags
        return TrainData(train_data, train_feats, train_tags, max_sent_len=self.max_sent_len, word2idx=self.word2idx)

    def get_val(self):
        refs = self.refs[1]
        return ValData(self.val_start_idx, self.val_end_idx, self.features, self.tags, refs)

    def get_test(self):
        refs = self.refs[2]
        return ValData(self.test_start_idx, self.test_end_idx, self.features, self.tags, refs)

    def split(self):
        return self.get_train(), self.get_val(), self.get_test()

    def decode(self, logsoftmax):
        outputs = []
        batch_size = logsoftmax.shape[0]
        argmax = logsoftmax.argmax(-1)
        for i in range(batch_size):
            seq = argmax[i]
            sentence = []
            for j in range(seq.shape[0]):
                if seq[j].item() == self.word2idx['<eos>'] or seq[j].item() == self.word2idx['<pad>']:
                    break
                sentence.append(self.idx2word[seq[j].item()])
            sentence = ' '.join(sentence)
            outputs.append(sentence)
        return outputs


class TrainData(Dataset):
    def __init__(self, data, feats, tags, max_sent_len, word2idx):
        self.data = data
        self.feats = feats
        self.tags = tags
        self.max_sent_len = max_sent_len
        self.word2idx = word2idx

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        caption = self.data[0][idx]
        video_index = self.data[1][idx]
        feat = self.feats[video_index]
        tags = self.tags[video_index]
        return feat, tags, caption


class ValData(Dataset):
    def __init__(self, start_idx, end_idx, feats, tags, refs):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.feats = feats
        self.tags = tags
        self.refs = refs

    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, idx):
        feat = self.feats[idx + self.start_idx]
        tags = self.tags[idx + self.start_idx]
        caption = self.refs[idx]
        return feat, tags, caption  # , caption


if __name__ == '__main__':
    dm = DataManager('data')
    train, val, test = dm.split()
    # print(train[0])
    # print(val[0])
    # print(test[0])
    train_loader = DataLoader(train, batch_size=10, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val, batch_size=5, collate_fn=collate_fn)
    for item in train_loader:
        break

    for item in val_loader:
        break
