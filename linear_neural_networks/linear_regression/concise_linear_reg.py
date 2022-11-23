"""concise implementation of linear regression using a deep leanring frame (pytorch)"""
import torch
import numpy as np
# from d21 import torch as d2l
from synthetic_dataset import synthetic_data
from torch.utils import data
from torch import nn


## genearte synthetic dataset using d2l lib..
def generate_syntheticData(w, b):
    """generate synthetic dataset for training a linear regression model
    
    args:
    w: torch.tensor
    b: float
    
    return 
    features: torch.tensor
    labels: torch.tensor"""
    features, labels = synthetic_data(w, b, 1000)
    return features, labels

def load_data(data_arrays, batch, is_train = True):
    """load the dataset into batches
    
    args:
    data_arrays: tuple(torch.tensor, torch.tensor) -> feautures and labels
    batch: int
    is_train: bool 
    
    return
    data iterator"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch, shuffle = is_train)

def create_model(w):
    """create simple linear regression model in pytorch uisng nn.Linear class
    
    args:
    w: torch.tensor
    
    return 
    net: nn.Linear"""

    in_shape = w.shape[0]
    out_shape = 1
    net = nn.Sequential(nn.Linear(in_shape, out_shape))
    return net

def train(model, optimizer, loss, data_iter, data, epochs=10):
    """train a simple linear regression model in pytorch
    
    args
    model: nn.sequentail
    optimizer: nn.optim.SGD
    loss: nn.MSELoss
    data_iter: data.DataLoader
    epochs: int"""
    features, labels = data
    for epoch in range(epochs):
        for X, y in data_iter:
            predictions = model(X)
            l = loss(y, predictions)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        epoch_loss = loss(model(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')




if __name__ == "__main__":
    w = torch.tensor([2, -3.4])
    b = 0
    batch_size = 32
    features, labels = generate_syntheticData(w, b)
    print(features.shape, labels.shape)

    #loading the dataset into batches
    data_iterator = load_data((features, labels), batch_size, True)
    print(next(iter(data_iterator)))

    #builindg training pipe line
    model = create_model(w)
    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr= 1e-2)

    #training
    train(model, optimizer, loss, data_iterator, (features, labels), 200)
    
    #Error measurement
    w_pred = model[0].weight.data
    print('error in estimating w:', w - w_pred.reshape(w.shape))
    b_pred = model[0].bias.data
    print('error in estimating b:', b - b_pred)

