import json
import os
import torch
import argparse
import numpy as np

from datasets import load_dataset
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer

from os.path import join, abspath, dirname

from modeling import LM

from vocab import *
from dataset import LAMADataset



SUPPORT_MODELS = ['bert-base-cased', 'bert-large-cased', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def construct_generation_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default='gpt2-medium', choices=SUPPORT_MODELS)
    parser.add_argument("--pseudo_token", type=str, default='[PROMPT]')



    parser.add_argument("--template", type=str, default="(3, 3, 3)")

    parser.add_argument("--seed", type=int, default=34, help="random seed for initialization")


    parser.add_argument("--vocab_strategy", type=str, default="shared", choices=['original', 'shared', 'lama'])
    parser.add_argument("--lstm_dropout", type=float, default=0.0)


    # directories
    parser.add_argument("--data_dir", type=str, default=join(abspath(dirname(__file__)), './data/LAMA'))
    parser.add_argument("--out_dir", type=str, default=join(abspath(dirname(__file__)), './out/LAMA'))


    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.template = eval(args.template) if type(args.template) is not tuple else args.template

    assert type(args.template) is tuple
    
    set_seed(args)

    return args

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = 'cuda:0'

        tokenizer_src = self.args.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, use_fast=False)
        init_vocab(args)
        
        self.test_set = LAMADataset('test', self.tokenizer, self.args)
        self.test_loader = DataLoader(self.test_set, batch_size=16)

        os.makedirs(self.get_save_path(), exist_ok=True)

        self.model = LM(args, self.device, self.args.template)

    def evaluate(self):
        self.model.eval()
        loader = self.test_loader
        dataset = self.test_set
        with torch.no_grad():
            self.model.eval()
            hit1, loss = 0, 0
            for x_hs, x_ts in loader:
                _loss, _hit1, top10 = self.model(x_hs, x_ts, return_candidates=True)
                hit1 += _hit1
                loss += _loss.item()
            hit1 /= len(dataset)
            print(len(dataset))
            print("Loss: {} Hit@1:".format(loss / len(dataset)), hit1)
        return loss, hit1
    
    def get_save_path(self):
        return join(self.args.out_dir, 'prompt_model', self.args.model_name, 'search')

    def get_checkpoint(self, test_hit1):
        ckpt_name = "test_{}.ckpt".format(round(test_hit1 * 100, 4))
        return {'test_hit@1': test_hit1,
                'test_size': len(self.test_set),
                'ckpt_name': ckpt_name,
                'time': datetime.now(),
                'args': self.args}

    def save(self, best_ckpt):
        ckpt_name = best_ckpt['ckpt_name']
        path = self.get_save_path()
        os.makedirs(path, exist_ok=True)
        torch.save(best_ckpt, join(path, ckpt_name))
        print("# Checkpoint {} saved.".format(ckpt_name))

    def train(self):
        best_dev, early_stop, has_adjusted = 0, 0, True
        best_ckpt = None

        test_loss, test_hit1 = self.evaluate()
        best_ckpt = self.get_checkpoint(test_hit1)
        early_stop = 0

        self.save(best_ckpt)
        return best_ckpt


def main():
    args = construct_generation_args()
    print(args.model_name)
    trainer = Trainer(args)
    trainer.train()

if __name__ == '__main__':
    main()



