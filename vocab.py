import json
from os.path import join

def init_vocab(args):
    global shared_vocab, lama_vocab
    shared_vocab = json.load(open(join(args.data_dir, '29k-vocab.json')))
    lama_vocab = json.load(open(join(args.data_dir, '34k-vocab.json')))


def get_vocab(model_name, strategy):
    if strategy == 'shared':
        assert model_name in shared_vocab
        return shared_vocab[model_name]
    elif strategy == 'lama':
        assert model_name in lama_vocab
        return lama_vocab[model_name]

def get_vocab_by_strategy(args, tokenizer):
    if args.vocab_strategy == 'original':
        return tokenizer.get_vocab()
    else:
        return get_vocab(args.model_name, args.vocab_strategy)



