from torch.utils.data import Dataset
import json

from vocab import get_vocab_by_strategy, token_wrapper


class LAMADataset(Dataset):
    def __init__(self, dataset_type, tokenizer, args, pid=None):
        super().__init__()
        self.args = args
        self.data = list()
        self.dataset_type = dataset_type
        self.x_hs, self.x_ts, self.x_rels, self.x_pids = [], [], [], []

        cases = []
        with open( f'/home/tjrals/jinseok/Prompting/test_data/{dataset_type}_original_relations.json', "r") as f:
            for line in f.readlines():
                cases.append(json.loads(line))

        vocab = get_vocab_by_strategy(args, tokenizer)
        for d in cases:
            if token_wrapper(args, d['obj_label']) not in vocab:
                continue
            if pid!=None:
                if d['predicate_id'] != pid:
                    continue
            self.x_hs.append(d['sub_label'])
            self.x_ts.append(d['obj_label'])
            self.x_rels.append(d['relation'])
            self.x_pids.append(d['predicate_id'])
            self.data.append(d)
        print(self.dataset_type, len(self.data))

    def __len__(self):
        return len(self.data)


    def __getitem__(self, i):
        return self.data[i]['sub_label'], self.data[i]['obj_label'], self.data[i]['relation'], self.data[i]['predicate_id']
