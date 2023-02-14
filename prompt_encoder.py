import torch
import torch.nn as nn

class PromptEncoder(torch.nn.Module):
    def __init__(self, template, hidden_size, tokenizer, device, args):
        super().__init__()
        self.device = device
        self.spell_length = sum(template)
        self.hidden_size = hidden_size
        self.tokenizer = tokenizer
        self.args = args

        # ent embedding
        self.cloze_length = template
        self.cloze_mask = [
            [1] * self.cloze_length[0]  # first cloze
            + [1] * self.cloze_length[1]  # second cloze
            + [1] * self.cloze_length[2]  # third cloze
        ]
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool().to(self.device) # tensor([[True, True, True, True, True, True, True, True, True]]) if bert
        # self.seq_indices = torch.LongTensor(list(range(len(self.cloze_mask[0])))).to(self.device) # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8]) if bert

        # embedding
        self.embedding = torch.nn.Embedding(42, self.hidden_size).to(self.device)

        # LSTM
        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size, # 768
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=self.args.lstm_dropout,
                                       bidirectional=True,
                                       batch_first=True)

        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size))

        print("init prompt encoder...")

    def forward(self, x_pids):
        ids = x_pids.tolist()
        if len(ids)!=8:
            ids += [0] * (8-len(x_pids))
        sequence = torch.LongTensor([0] + ids).to(self.device) # tensor([0] + [ids padding with 0]) if bert
        input_embeds = self.embedding(sequence).unsqueeze(0) # torch.Size([1, 9, 768])
        lstm_out = self.lstm_head(input_embeds)[0]
        output_embeds = self.mlp_head(lstm_out).squeeze()
        return output_embeds
                        



