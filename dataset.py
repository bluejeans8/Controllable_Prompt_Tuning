from torch.utils.data import Dataset
import json

from vocab import get_vocab_by_strategy, token_wrapper


class LAMADataset(Dataset):
    def __init__(self, dataset_type, tokenizer, args):
        super().__init__()
        self.args = args
        self.data = list()
        self.dataset_type = dataset_type
        self.x_hs, self.x_ts = [], []

        data = []
        with open( "/home/tjrals/jinseok/js_p-tuning/test_data/test_original_rob_relations.json", "r") as f:
            for line in f.readlines():
                data.append(json.loads(line))


        vocab = get_vocab_by_strategy(args, tokenizer)
        for d in data:
            if token_wrapper(args, d['obj_label']) not in vocab:
                continue
            self.x_ts.append(d['obj_label'])
            self.x_hs.append(d['masked_sentence'])
            self.data.append(d)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, i):
        return self.data[i]['masked_sentence'], self.data[i]['obj_label']
