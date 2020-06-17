import torch
import torch.nn as nn


class NLLLossWithLength(nn.Module):
    def __init__(self, ignore_index, beta=0.7):
        super(NLLLossWithLength, self).__init__()
        self.ignore_index = ignore_index
        self.beta = beta
        self.nllloss = nn.NLLLoss(reduction='none', ignore_index=self.ignore_index)

    def forward(self, out, gt):
        bs = out.shape[0]
        max_len = out.shape[1]
        sentence_len = (gt != self.ignore_index).sum(dim=-1).view(bs,1)
        sentence_len = sentence_len.repeat(1,gt.shape[-1])
        # print('init sentence len')
        # print(sentence_len)
        out = out.view(-1, out.shape[-1])
        gt = gt.view(-1)
        sentence_len = sentence_len.view(-1)
        weight = 1 / sentence_len.float() ** self.beta
        # print('sentence_len')
        # print(sentence_len)
        loss = self.nllloss(out, gt)
        # print('unweighted loss')
        # print(loss)
        # print(loss.shape)
        # print('origin loss')
        # print(loss)
        # print('sentence_len')
        # print(sentence_len)
        # print('weight')
        # print(weight)
        # print(weight.shape)
        loss = loss * weight
        # print('weighted loss')
        # print(loss)
        # print('res loss')
        # print(loss)
        # print('sentence loss')
        # print('avg loss')
        # print(torch.sum(loss) / bs)
        # assert False
        return torch.sum(loss) / bs
