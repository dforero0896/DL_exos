# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch
from torch import Tensor
import os
import sys
sys.path.append("..")
import dlc_practical_prologue as prologue
os.environ["PYTORCH_DATA_DIR"]="data"


# %%
train_input, train_target, test_input, test_target = prologue.load_data()


# %%
def nearest_classification(train_input, train_target, x):
    distances = (train_input - x[None,:]).pow(2).mean(axis=-1)
    return train_target[torch.argmin(distances)]


print(nearest_classification(train_input, train_target, test_input[0]))


# %%
def compute_nb_errors(train_input, train_target, test_input, test_target, mean=None, proj=None):
    ntest = test_input.shape[0]
    prediction = torch.empty(ntest)
    accuracy = torch.empty(ntest)
    if mean is not None:
        train_input-=mean
        test_input-=mean
    if proj is not None:
        train_input=train_input.mm(proj.t())
        test_input=test_input.mm(proj.t())
    for i in range(ntest):
        prediction[i] = nearest_classification(train_input, train_target, test_input[i])
    accuracy = prediction!=test_target
    return accuracy.int().sum(), (accuracy).float().mean()


# %%
def PCA(x):
    
    mean = x.mean(axis=0)
    std_x = x - mean
    cov = std_x.t().mm(std_x)
    eival, eivect = torch.eig(cov, eigenvectors=True)
    _,  sorter = torch.sort(eival[:,0], descending=True)
    return mean, eivect[:,sorter].t()



# %%
torch.manual_seed(42)
# Modified from solutions
for c in [ False, True ]: # To compare MNIST and CIFAR

    train_input, train_target, test_input, test_target = prologue.load_data(cifar = c)

    # Original test on dataset. Define baseline

    nb_errors, perc_errors = compute_nb_errors(train_input, train_target, test_input, test_target)
    print(f'Original errors: nb_errors {nb_errors} error {100*perc_errors:.02f}%')

    # Project onto 100d random basis, that is a set of d 100-elem long vectors. 
    # the basis defines the projection matrix.

    basis = train_input.new(100, train_input.size(1)).normal_()

    nb_errors, perc_errors = compute_nb_errors(train_input, train_target, test_input, test_target, None, basis)
    print(f'Random basis: nb_errors {nb_errors} error {100*perc_errors:.02f}%')

    # Project onto PCA space with the first npc PCs taken into account

    mean, basis = PCA(train_input)
    
    for npc in [ 100, 50, 10, 3 ]:
        nb_errors, perc_errors = compute_nb_errors(train_input, train_target, test_input, test_target, mean, basis[:npc])
        print(f'PCA {npc:d}d nb_errors {nb_errors} error {100*perc_errors:.02f}%')


# %%



