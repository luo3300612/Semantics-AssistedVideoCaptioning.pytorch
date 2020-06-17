import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SemanticLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, feat_size, tag_size, hidden_size, bos_idx=0, eos_idx=1, padding_idx=2,
                 embedding_array=None, freeze_embedding=True, schedule_sample_prob=0, schedule_sample_method='greedy'):
        super(SemanticLSTM, self).__init__()
        self.feat_size = feat_size
        self.tag_size = tag_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_dropout = nn.Dropout(p=0.5)
        self.input_size = feat_size + tag_size + embedding_dim
        self.hidden_size = hidden_size
        self.schedule_sample_prob = schedule_sample_prob
        self.schedule_sample_method = schedule_sample_method

        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.padding_idx = padding_idx
        if embedding_array is not None:
            embedding_array = torch.FloatTensor(embedding_array)
            embedding_array = torch.cat([embedding_array, torch.zeros((2, embedding_dim))], dim=0)  # for bos and pad
            self.word_embed = nn.Embedding.from_pretrained(embedding_array, freeze=freeze_embedding)
        else:
            self.word_embed = nn.Embedding(vocab_size, embedding_dim)
        self.feat2input = nn.Linear(self.feat_size, self.embedding_dim, bias=False)

        self.feat_dropout = nn.Dropout(p=0.5)
        self.tag_dropout = nn.Dropout(p=0.5)

        self.tag2lstm1 = nn.Linear(self.tag_size, 4 * self.hidden_size, bias=False)
        self.feat2lstm = nn.Linear(self.feat_size, 4 * self.hidden_size, bias=False)
        self.tag2lstm2 = nn.Linear(self.tag_size, 4 * self.hidden_size, bias=False)

        self.word2lstm = nn.Linear(self.embedding_dim, 4 * self.hidden_size, bias=False)

        self.fc_i = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.fc_f = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.fc_o = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.fc_c = nn.Linear(2 * self.hidden_size, self.hidden_size)

        self.fc_hidden_state_i = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_hidden_state_f = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_hidden_state_o = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_hidden_state_c = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_tag_i = nn.Linear(self.tag_size, self.hidden_size, bias=False)
        self.fc_tag_f = nn.Linear(self.tag_size, self.hidden_size, bias=False)
        self.fc_tag_o = nn.Linear(self.tag_size, self.hidden_size, bias=False)
        self.fc_tag_c = nn.Linear(self.tag_size, self.hidden_size, bias=False)
        self.fc_both_i = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_both_f = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_both_o = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_both_c = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.h_dropout = nn.Dropout(0.5)
        self.word2logit = nn.Linear(self.embedding_dim, self.hidden_size)
        self.logit_bias = nn.Parameter(torch.zeros((vocab_size - 2,)), requires_grad=True)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def init_state(self, x):
        bs = x.shape[0]
        return (torch.zeros((bs, self.hidden_size), device=x.device),
                torch.zeros((bs, self.hidden_size), device=x.device))

    def core(self, word_embedding, feats, tags, tmps, state):
        tmp2_i, tmp2_f, tmp2_o, tmp2_c = torch.split(tmps['tmp2'], self.hidden_size, dim=-1)
        tmp3_i, tmp3_f, tmp3_o, tmp3_c = torch.split(tmps['tmp3'], self.hidden_size, dim=-1)
        tmp4_i, tmp4_f, tmp4_o, tmp4_c = torch.split(tmps['tmp4'], self.hidden_size, dim=-1)

        tmp1_i, tmp1_f, tmp1_o, tmp1_c = torch.split(self.word2lstm(word_embedding), self.hidden_size, dim=-1)

        tmp_i = torch.cat([tmp1_i * tmp2_i, tmp3_i * tmp4_i], dim=-1)
        tmp_f = torch.cat([tmp1_f * tmp2_f, tmp3_f * tmp4_f], dim=-1)
        tmp_o = torch.cat([tmp1_o * tmp2_o, tmp3_o * tmp4_o], dim=-1)
        tmp_c = torch.cat([tmp1_c * tmp2_c, tmp3_c * tmp4_c], dim=-1)
        input_i = self.fc_i(tmp_i)
        input_f = self.fc_f(tmp_f)
        input_o = self.fc_o(tmp_o)
        input_c = self.fc_c(tmp_c)

        preact_i = self.fc_both_i(self.fc_hidden_state_i(state[0]) * self.fc_tag_i(tags)) + input_i
        preact_f = self.fc_both_f(self.fc_hidden_state_f(state[0]) * self.fc_tag_f(tags)) + input_f
        preact_o = self.fc_both_o(self.fc_hidden_state_o(state[0]) * self.fc_tag_o(tags)) + input_o
        preact_c = self.fc_both_c(self.fc_hidden_state_c(state[0]) * self.fc_tag_c(tags)) + input_c

        i = torch.sigmoid(preact_i)
        f = torch.sigmoid(preact_f)
        o = torch.sigmoid(preact_o)
        c = torch.tanh(preact_c)

        c = f * state[1] + i * c
        h = o * torch.tanh(c)
        return (h, c)

    def prepare_feats(self, feats, tags, seq):
        feats = self.feat_dropout(feats)
        tags = self.tag_dropout(tags)

        tmps = {}
        tmps['tmp2'] = self.tag2lstm1(tags)
        tmps['tmp3'] = self.feat2lstm(feats)
        tmps['tmp4'] = self.tag2lstm2(tags)
        return feats, tags, seq, tmps

    def forward(self, feats, tags, seq):
        state = self.init_state(feats)
        bs = feats.shape[0]
        outputs = []

        feats, tags, seq, tmps = self.prepare_feats(feats, tags, seq)
        # actually we do not use bos token, we use visual feats instead
        # this is just for put all code into a single loop
        bos = torch.ones(bs, device=feats.device).long() * self.bos_idx
        seq = torch.cat([bos.unsqueeze(1), seq], dim=1)
        # use visual feats at first step
        for i in range(seq.shape[1]):
            rand = np.random.uniform(0, 1, (bs,))
            if (seq[:, i] == self.eos_idx).sum() + (seq[:, i] == self.padding_idx).sum() == bs:
                break
            if i is 0:  # start token
                word_embedding = self.feat2input(feats)
            elif self.schedule_sample_prob != 0 and (
                    rand < self.schedule_sample_prob).any():  # schedula sample
                xt = seq[:, i].data.clone()
                index = rand < self.schedule_sample_prob
                last_output = outputs[-1].detach()
                if self.schedule_sample_method == 'greedy':
                    words = last_output.argmax(-1)
                elif self.schedule_sample_method == 'multinomial':
                    distribution = torch.exp(last_output)
                    words = torch.multinomial(distribution, 1).squeeze(-1)
                else:
                    raise NotImplementedError
                xt[index] = words[index]
                word_embedding = self.embedding_dropout(self.word_embed(xt))
            else:  # Teacher Forcings
                word_embedding = self.embedding_dropout(self.word_embed(seq[:, i]))
            state = self.core(word_embedding, feats, tags, tmps, state)
            logit_weight = torch.matmul(self.word2logit.weight, self.word_embed.weight[:-2, :].T)
            logit = F.linear(self.h_dropout(state[0]), logit_weight.T, self.logit_bias)
            outputs.append(logit)
        res = torch.stack(outputs, dim=1)
        res = F.log_softmax(res, dim=-1)
        return res

    def sample(self, feats, tags, maxlen, mode='greedy'):
        state = self.init_state(feats)
        outputs = []
        bs = feats.shape[0]

        feats, tags, seq, tmps = self.prepare_feats(feats, tags, None)
        # bos = torch.ones(bs, device=feats.device).long() * self.bos_idx
        is_finished = torch.zeros(bs)
        for i in range(maxlen + 1):
            if i is 0:  # use visual feats at first step
                word_embedding = self.feat2input(feats)
            elif mode == 'greedy':
                last_output = outputs[-1]
                last_token = last_output.argmax(-1)
                is_finished[last_token == self.eos_idx] = 1
                if is_finished.sum() == bs:  # all finished sample
                    break
                word_embedding = self.embedding_dropout(self.word_embed(last_token))
            else:
                raise NotImplementedError

            state = self.core(word_embedding, feats, tags, tmps, state)
            logit_weight = torch.matmul(self.word2logit.weight, self.word_embed.weight[:-2, :].T)
            logit = F.linear(self.h_dropout(state[0]), logit_weight.T, self.logit_bias)
            outputs.append(logit)
        res = torch.stack(outputs, dim=1)
        res = F.log_softmax(res, dim=-1)
        return res


if __name__ == '__main__':
    model = SemanticLSTM(10, 20, 120, 100)
    feats = torch.randn(16, 50)
    tags = torch.randn(16, 50)
    seq = torch.randint(0, 10, (16, 10))
    out = model(feats, tags, seq)
    print(out.shape)
    sample_out = model.sample(feats, tags, 20)
    print(sample_out.shape)
