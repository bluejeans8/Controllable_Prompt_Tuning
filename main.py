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

import glob
import re



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

    parser.add_argument("--use_original_template", type=bool, default=False)

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
    def __init__(self, args, pid=None):
        self.args = args
        self.pid = pid
        self.device = 'cuda:0'

        tokenizer_src = self.args.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, use_fast=False)
        init_vocab(args)
        
        self.train_set = LAMADataset('train', self.tokenizer, self.args, self.pid)
        self.train_loader = DataLoader(self.train_set, batch_size=8, shuffle=True, drop_last=True)

        self.dev_set = LAMADataset('dev', self.tokenizer, self.args, self.pid)
        self.dev_loader = DataLoader(self.dev_set, batch_size=8, shuffle=True)

        self.test_set = LAMADataset('test', self.tokenizer, self.args, self.pid)
        self.test_loader = DataLoader(self.test_set, batch_size=8, shuffle=True)

        os.makedirs(self.get_save_path(), exist_ok=True)

        self.model = LM(args, self.device, self.args.template)

    def evaluate(self, epoch_idx, dataset_type):
        self.model.eval()
        if dataset_type == 'test':
            loader = self.test_loader
            dataset = self.test_set
        else:
            loader = self.dev_loader
            dataset = self.dev_set

        with torch.no_grad():
            self.model.eval()
            hit1, loss = 0, 0
            for x_hs, x_ts, x_rels, x_pids in loader:
                _loss, _hit1, top10 = self.model(x_hs, x_ts, x_rels, x_pids, return_candidates=True)
                hit1 += _hit1
                loss += _loss.item()
            hit1 /= len(dataset)
            print("{} {} Epoch Loss: {} Hit@1:".format(dataset_type, epoch_idx, loss / len(dataset)), hit1)
        return loss, hit1
    
    def get_save_path(self):
        return join(self.args.out_dir, 'prompt_model', self.args.model_name, 'search')

    def get_checkpoint(self, dev_hit1, test_hit1):
        ckpt_name = "{} dev_{}_test_{}.ckpt".format(self.pid, round(dev_hit1 * 100, 4), round(test_hit1 * 100, 4))
        return {'dev_hit@1': dev_hit1,
                'test_hit@1': test_hit1,
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
        params = [{'params': self.model.prompt_encoder.parameters()}]
        optimizer = torch.optim.Adam(params, lr=1e-5, weight_decay=0.0005)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98) 

        if self.args.use_original_template == True:
                test_loss, test_hit1 = self.evaluate(0, 'test')
                dev_loss, dev_hit1 = self.evaluate(0, 'dev')
                best_ckpt = self.get_checkpoint(dev_hit1, test_hit1)
                self.save(best_ckpt)
                return best_ckpt

        for epoch_idx in range(100):
            dev_loss, dev_hit1 = self.evaluate(epoch_idx, 'dev')
            if dev_hit1 >= best_dev:
                test_loss, test_hit1 = self.evaluate(epoch_idx, 'test')
                best_ckpt = self.get_checkpoint(dev_hit1, test_hit1)
                best_dev = dev_hit1
                early_stop = 0
            else:
                early_stop += 1
                if early_stop >= 20:
                    self.save(best_ckpt)
                    print("Early stopping at epoch {}.".format(epoch_idx))
                    return best_ckpt                    

            hit1, num_of_samples = 0, 0
            tot_loss = 0
            for batch_idx, batch in tqdm(enumerate(self.train_loader)):
                self.model.train()
                loss, batch_hit1 = self.model(batch[0], batch[1], batch[2], batch[3])
                hit1 += batch_hit1
                tot_loss += loss.item()
                num_of_samples += len(batch[0])

                loss.backward()
                torch.cuda.empty_cache()
                optimizer.step()
                torch.cuda.empty_cache()
                optimizer.zero_grad()
            my_lr_scheduler.step()

        self.save(best_ckpt)
        return best_ckpt


def train_by_relation(args):
    pids = [pid for pid in range(1,42)]
    print(pids)
    with open("result_gpt2_medium.txt", "w") as rf:
        total_size = 0
        total_hits = 0
        for pid in pids:
            print("pid:", pid)
            trainer = Trainer(args, pid)
            best_ckpt = trainer.train()
            size = best_ckpt['test_size']
            hits = best_ckpt['test_hit@1'] * size
            print("cur:", hits, size, hits/size)

            total_size += size
            total_hits += hits
            print("acc:", total_hits, total_size,  total_hits/total_size)
            rf.write(f"{pid}, {hits}, {size}, {hits/size}, {total_hits}, {total_size}, {total_hits/total_size}\n")
        
        print("result:", total_hits, total_size, total_hits/total_size)

def train_whole(args):
    trainer = Trainer(args)
    best_ckpt = trainer.train()
    size = best_ckpt['test_size']
    hits = best_ckpt['test_hit@1'] * size
    print("result:", hits, size, hits/size)
       


def main():
    args = construct_generation_args()
    print(args.model_name)
    train_by_relation(args)

if __name__ == '__main__':
    main()



