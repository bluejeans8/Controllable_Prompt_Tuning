import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForMaskedLM, GPT2LMHeadModel

from vocab import *
import re

from prompt_encoder import PromptEncoder

class LM(torch.nn.Module):

    def __init__(self, args, device, template):
        super().__init__()
        self.args = args
        self.device = device

        tokenizer_src = self.args.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, use_fast=False)

        if 'bert' in self.args.model_name:
            MODEL_CLASS = AutoModelForMaskedLM
        elif 'gpt' in self.args.model_name:
            MODEL_CLASS = GPT2LMHeadModel
        self.model = MODEL_CLASS.from_pretrained(self.args.model_name).to(self.device)

        if 'bert' in self.args.model_name:
            self.embeddings = self.model.bert.get_input_embeddings()
        elif 'gpt' in self.args.model_name:
            self.embeddings = self.model.base_model.get_input_embeddings()

        
        # set allowed vocab set
        self.vocab = self.tokenizer.get_vocab()
        self.allowed_vocab_ids = set(self.vocab[k] for k in get_vocab_by_strategy(self.args, self.tokenizer))

        if 'gpt' in self.args.model_name:
            template = (template[0], template[1], 0)
            print(template)
        self.template = template
        

        # load prompt encoder
        self.hidden_size = self.embeddings.embedding_dim # 768
        self.tokenizer.add_special_tokens({'additional_special_tokens': [self.args.pseudo_token]})
        self.pseudo_token_id = self.tokenizer.get_vocab()[self.args.pseudo_token]

        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id


        self.spell_length = sum(self.template)
        self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, self.tokenizer, self.device, args)
        self.prompt_encoder = self.prompt_encoder.to(self.device)


    def embed_input(self, queries, x_pids):
        bz = queries.shape[0]
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.tokenizer.unk_token_id
        raw_embeds = self.embeddings(queries_for_embedding)

        if self.args.use_original_template:
            return raw_embeds

        blocked_indices = (queries == self.pseudo_token_id).nonzero().reshape((bz, self.spell_length, 2))[:, :, 1]  # bz
        replace_embeds = self.prompt_encoder(x_pids)
        for bidx in range(bz):
            for i in range(self.prompt_encoder.spell_length):
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
        return raw_embeds


    def get_query(self, x_h, x_rel, prompt_tokens):
        # For using handcraft prompts
        if self.args.use_original_template:
            if 'gpt' in self.args.model_name:
                query = re.sub(r'\[Y\].*', '', x_rel).replace('[X]', x_h)
            else:
                query = x_rel.replace('[Y]', self.tokenizer.mask_token).replace('[X]', x_h)
            return self.tokenizer(' ' + query)['input_ids']
        
        # For P-tuning
        if 'bert' in self.args.model_name:
            return [[self.tokenizer.cls_token_id]  # [CLS]
                    + prompt_tokens * self.template[0]
                    + [self.tokenizer.mask_token_id]  # head entity
                    + prompt_tokens * self.template[1]
                    + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + x_h))  # [MASK] (tail entity)
                    + (prompt_tokens * self.template[2] if self.template[2] > 0 else self.tokenizer.convert_tokens_to_ids(['.']))
                    + [self.tokenizer.sep_token_id]
                    ]
        
        

    def forward(self, x_hs, x_ts, x_rels, x_pids, return_candidates=False):
        bz = len(x_hs)

        prompt_tokens = [self.pseudo_token_id]
        x_ts = [token_wrapper(self.args, x_t) for x_t in x_ts]
        queries = [torch.LongTensor(self.get_query(x_hs[i], x_rels[i], prompt_tokens)).squeeze(0) for i in range(bz)]
        queries = pad_sequence(queries, True, padding_value=self.pad_token_id).long().to(self.device)

        label_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(x_ts)).reshape((bz, -1)).to(self.device)
        attention_mask = queries != self.pad_token_id

        inputs_embeds = self.embed_input(queries, x_pids)

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

        def gpt_out():
            labels = torch.empty_like(queries).fill_(-100).long().to(self.device)  # bz * seq_len
            label_mask = (attention_mask.long().sum(dim=1) - 1).unsqueeze(1).to(self.device)
            labels = labels.scatter_(1, label_mask, label_ids)

            output = self.model(inputs_embeds=inputs_embeds.to(self.device).half(),
                                attention_mask=attention_mask.to(self.device).half(),
                                labels=labels.to(self.device))
            loss, logits = output.loss, output.logits

            pred_ids = torch.argsort(logits, dim=2, descending=True)
            hit1 = 0
            top10 = []
            for i in range(bz):
                top10.append([])
                pred_seq = pred_ids[i, label_mask[i, 0]].tolist()
                for pred in pred_seq:
                    if pred in self.allowed_vocab_ids:
                        top10[-1].append(pred)
                        if len(top10[-1]) >= 10:
                            break
                pred = top10[-1][0]
                if pred == label_ids[i, 0]:
                    hit1 += 1

            if return_candidates:
                return loss, hit1, top10
            return loss, hit1

        if 'bert' in self.args.model_name:
            return bert_out()
        elif 'gpt' in self.args.model_name:
            return gpt_out()

