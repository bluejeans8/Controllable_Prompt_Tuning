import torch
import numpy as np
from sklearn.manifold import TSNE
from prompt_encoder import PromptEncoder

PATH = "/home/tjrals/jinseok/Prompting/out/LAMA/prompt_model/bert-base-cased/search/Model None dev_55.082_test_52.1343.ckpt"

checkpoint = torch.load(PATH)
model = checkpoint['prompt_encoder']
model.eval()

data = model(1)[0].unsqueeze(0)
print(data.shape)

for i in range(2,42):
    data = torch.cat((data, model(i)[0].unsqueeze(0)),0)
print(data.shape)

n_components = 2
t = TSNE(n_components=n_components)
tsne = t.fit_transform(data.detach().cpu())

print(tsne.shape)

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))
 
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
 
    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range
 
# extract x and y coordinates representing the positions of the images on T-SNE plot
tx = tsne[:, 0]
ty = tsne[:, 1]
 
tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)


# initialize a matplotlib plot
fig = plt.figure()
ax = fig.add_subplot(111)
 
# for every class, we'll add a scatter plot separately
for label in colors_per_class:
    # find the samples of the current class in the data
    indices = [i for i, l in enumerate(labels) if l == label]
 
    # extract the coordinates of the points of this class only
    current_tx = np.take(tx, indices)
    current_ty = np.take(ty, indices)
 
    # convert the class color to matplotlib format
    color = np.array(colors_per_class[label], dtype=np.float) / 255
 
    # add a scatter plot with the corresponding color and label
    ax.scatter(current_tx, current_ty, c=color, label=label)
 
# build a legend using the labels we set previously
ax.legend(loc='best')
 
# finally, show the plot
plt.show()