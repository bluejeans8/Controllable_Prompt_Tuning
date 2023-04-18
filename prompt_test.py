import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from prompt_encoder import PromptEncoder
import seaborn as sns

PATH = "/home/tjrals/jinseok/Prompting/out/LAMA/prompt_model/bert-base-cased/search/Model None dev_55.082_test_52.1343.ckpt"

checkpoint = torch.load(PATH)
model = checkpoint['prompt_encoder']
model.eval()

rms = 0
for i in range(0,9):
    rms += model(1)[i].mul(model(1)[i])
rms = torch.sqrt(rms)


data = rms.unsqueeze(0)

for i in range(2,42):
    rms = 0
    for j in range(0,9):
        rms += model(i)[j].mul(model(i)[j])
    rms = torch.sqrt(rms)
    data = torch.cat((data, rms.unsqueeze(0)),0)
print(data)

n_components = 2
t = TSNE(n_components=n_components)
tsne = t.fit_transform(data.detach().cpu())

print(tsne.shape)

labels = [i for i in range(1,42)]
fig = plt.figure(figsize = (10,10))
# plt.axis('off')
sns.set_style('darkgrid')
sns.scatterplot(x=tsne[:,0], y=tsne[:,1], hue=labels, legend='full', palette=sns.color_palette("bright", 41))
# plt.legend([i for i in range(1,42)])
for i in range(0,41):
    plt.text(x=tsne[i,0]+0.02, y=tsne[i,1]+0.02,s=i+1)
plt.savefig('abcde')
