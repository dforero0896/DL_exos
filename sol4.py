#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>
# Modified by Daniel Forero-Sánchez

import torch
from torch import nn
from torch.nn import functional as F

import dlc_practical_prologue as prologue

if torch.cuda.is_available():
    device='cuda'
else:
    device='cpu'


######################################################################

class Net(nn.Module):
    def __init__(self, n_hidden=200):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(256, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=3))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x

class Net2(nn.Module):
    def __init__(self, n_hidden=200):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(128*18*18, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=1))
        
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=1))
        
        x = F.relu(F.max_pool2d(self.conv3(x), kernel_size=2, stride=1))
        
        
        x = F.relu(self.fc1(x.view(-1, 128*18*18)))
        x = self.fc2(x)
        return x


def train_model(model, train_input, train_target, mini_batch_size,
                eta = 1e-1, nb_epochs = 100, criterion=nn.MSELoss(),
                verbose=True):

    for e in range(nb_epochs):
        acc_loss = 0
        # We do this with mini-batches
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            acc_loss = acc_loss + loss.item()

            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= eta * p.grad

        if verbose: print(e, acc_loss)

def compute_nb_errors(model, input, target, mini_batch_size):
    nb_errors = 0
    with torch.no_grad():
        for b in range(0, input.size(0), mini_batch_size):
            prediction = model(input.narrow(0, b, mini_batch_size))
            predicted_class = torch.argmax(prediction, axis=1)
            target_class = torch.argmax(target.narrow(0, b, mini_batch_size), axis=1)
            accuracy = (predicted_class == target_class)
            nb_errors += (~accuracy).int().sum().item()
    error_fraction = nb_errors / target.shape[0]
    return nb_errors, error_fraction



if __name__ == '__main__':

    

    
    eta, mini_batch_size = 1e-1, 100
    nb_epochs = 100


        
    model, criterion = Net(), nn.MSELoss()

    model = model.to(device)
    criterion = criterion.to(device)

    # Import data
    train_input, train_target, test_input, test_target = \
        prologue.load_data(one_hot_labels = True, normalize = True, flatten = False)

    train_input = train_input.to(device)
    train_target = train_target.to(device)
    test_input = test_input.to(device)
    test_target = test_target.to(device)

    train_model(model, train_input, train_target, mini_batch_size, eta, nb_epochs, criterion)

    nb_errors, error_fraction = compute_nb_errors(model, test_input, test_target, mini_batch_size)

    print(f"Error fraction = {error_fraction}")




    for n_hidden in [10, 50, 200, 500, 1000]:

        model = Net(n_hidden=n_hidden).to(device)
        criterion = nn.MSELoss().to(device)

        train_model(model, train_input, train_target, mini_batch_size, eta, nb_epochs, criterion, verbose=False)

        nb_errors, error_fraction = compute_nb_errors(model, test_input, test_target, mini_batch_size)

        print(f"N hidden units: {n_hidden}. Error fraction = {error_fraction}")



    model = Net2().to(device)
    criterion = nn.MSELoss().to(device)

    train_model(model, train_input, train_target, mini_batch_size, eta, nb_epochs, criterion, verbose=True)

    nb_errors, error_fraction = compute_nb_errors(model, test_input, test_target, mini_batch_size)

    print(f"Using 3 conv layers Error fraction = {error_fraction}")

    