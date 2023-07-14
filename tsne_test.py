import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from prompt_encoder import PromptEncoder
import pandas as pd

from transformers import AutoTokenizer, AutoModelForMaskedLM
from dataset import LAMADataset
from torch.utils.data import DataLoader
import argparse
from os.path import join, abspath, dirname
import json
from vocab import init_vocab



import seaborn as sns

# PATH = "/home/tjrals/jinseok/Prompting/out/LAMA/prompt_model/bert-base-cased/search/Model None dev_55.082_test_52.1343.ckpt"

# checkpoint = torch.load(PATH)
# model = checkpoint['prompt_encoder']
# model.eval()


# tokenizer_src = 'bert-base-cased'
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, use_fast=False)

# llm_model = AutoModelForMaskedLM.from_pretrained(tokenizer_src).to('cuda:0')
# embeddings = llm_model.bert.get_input_embeddings().to('cuda:0')


# parser = argparse.ArgumentParser()
# parser.add_argument("--model_name", type=str, default='bert-base-cased')
# parser.add_argument("--vocab_strategy", type=str, default="shared", choices=['original', 'shared', 'lama'])
# parser.add_argument("--data_dir", type=str, default=join(abspath(dirname(__file__)), './data/LAMA'))
# args = parser.parse_args()
# init_vocab(args)

# test_set = LAMADataset('test', tokenizer, args, None)



# rms_values = []

# for i in range(1,42):
#     rms = 0
#     for j in range(0,9):
#         rms += model(i)[j].mul(model(i)[j])
#     rms_values.append(rms)

# cnt = 0
# data = 0
# for case in test_set:
#     pid = case[3]
#     tk = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' ' + case[0]))
#     tk = torch.LongTensor(tk).squeeze(0).to('cuda:0')
#     em = embeddings(tk).to('cuda:0')
#     val = rms_values[pid-1]
#     for i in range(em.shape[0]):
#         val+= em[i].mul(em[i])
#     val /= (9 + em.shape[0])
#     val = torch.sqrt(val)
#     if cnt==0:
#         val = val.detach().cpu().numpy()
#         data = pd.DataFrame(val).transpose().assign(pid=[pid])
#     else:
#         # data = torch.cat((data, val.unsqueeze(0)),0)
#         val = val.detach().cpu().numpy()
#         data = pd.concat([data, pd.DataFrame(val).transpose().assign(pid=[pid])], ignore_index=True)
#     cnt+=1
#     if cnt%1000==0:
#         print(data)
#         print(cnt)

# n_components = 2
# t = TSNE(n_components=n_components)
# tsne = t.fit_transform(data[[i for i in range(0,768)]])

# print(tsne.shape)
# tsne_df = pd.DataFrame(tsne, columns = ['x', 'y'])
# tsne_df['pid'] = data['pid']

# tsne_df.to_csv('output.csv', index=False)

tsne_df = pd.read_csv('output.csv')
# print(tsne_df)


tsne_dfs = []
for pid in range(1,42):
    tsne_dfs.append(tsne_df[tsne_df['pid'] == pid])

labels = [i for i in range(1,42)]
for tdf in tsne_dfs:
    plt.scatter(tdf['x'], tdf['y'], s=0.1, label=tdf['pid'])

plt.xlabel('x')
plt.ylabel('y')
plt.legend(labels, fontsize="5")

# labels = [i for i in range(1,42)]
# fig = plt.figure(figsize = (10,10))
# # plt.axis('off')
# sns.set_style('darkgrid')
# sns.scatterplot(x=tsne[:,0], y=tsne[:,1], hue=labels, legend='full', palette=sns.color_palette("bright", 41))
# # plt.legend([i for i in range(1,42)])
# for i in range(0,41):
#     plt.text(x=tsne[i,0]+0.02, y=tsne[i,1]+0.02,s=i+1)
plt.savefig('abcdefg')
