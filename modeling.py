import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForMaskedLM

from vocab import *

class Bert(torch.nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device

        tokenizer_src = self.args.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, use_fast=False)

        self.model = AutoModelForMaskedLM.from_pretrained(self.args.model_name)
        self.model = self.model.to(self.device)

        self.embeddings = self.model.bert.get_input_embeddings()
        
        # set allowed vocab set
        self.vocab = self.tokenizer.get_vocab()
        self.allowed_vocab_ids = set(self.vocab[k] for k in get_vocab_by_strategy(self.args, self.tokenizer))

        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id


    def embed_input(self, queries):
        bz = queries.shape[0]
        queries_for_embedding = queries.clone()
        raw_embeds = self.embeddings(queries_for_embedding)

        return raw_embeds

    def get_query(self, x_h):
        query = x_h.replace('[MASK]', self.tokenizer.mask_token)
        return self.tokenizer(' ' + query)['input_ids']

    def forward(self, x_hs, x_ts, return_candidates=False):
        bz = len(x_hs)
        queries = [torch.LongTensor(self.get_query(x_hs[i])).squeeze(0) for i in range(bz)]
        queries = pad_sequence(queries, True, padding_value=self.pad_token_id).long().to(self.device)

        label_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(x_ts)).reshape((bz, -1)).to(self.device)
        attention_mask = queries != self.pad_token_id

        inputs_embeds = self.embed_input(queries)

        def bert_out():
            label_mask = (queries == self.tokenizer.mask_token_id).nonzero().reshape(bz, -1)[:, 1].unsqueeze(1).to(self.device)  # bz * 1
            labels = torch.empty_like(queries).fill_(-100).long().to(self.device)  # bz * seq_len
            labels = labels.scatter_(1, label_mask, label_ids)
            output = self.model(inputs_embeds=inputs_embeds.to(self.device), attention_mask=attention_mask.to(self.device).bool(), labels=labels.to(self.device))
            loss, logits = output.loss, output.logits

            pred_ids = torch.argsort(logits, dim=2, descending=True)
            hit1 = 0
            top10 = []
            for i in range(bz):
                pred_seq = pred_ids[i, label_mask[i, 0]].tolist()
                for pred in pred_seq:
                    if pred in self.allowed_vocab_ids:
                        break
                if pred == label_ids[i, 0]:
                    hit1 += 1

            if return_candidates:
                return loss, hit1, top10
            return loss, hit1

        return bert_out()


