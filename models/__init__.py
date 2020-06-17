from .SemanticLSTM import SemanticLSTM


def init_model(args, dm):
    if args.model == 'standard':
        model = SemanticLSTM(len(dm.idx2word), args.embedding_dim, 1536 + 2048, 300,
                             args.hidden_dim,
                             bos_idx=dm.word2idx['<bos>'],
                             eos_idx=dm.word2idx['<eos>'],
                             padding_idx=dm.word2idx['<pad>'],
                             embedding_array=dm.corpus[5],
                             schedule_sample_prob=args.schedule_sample_prob,
                             schedule_sample_method=args.schedule_sample_method)
    else:
        raise NotImplementedError
    return model
