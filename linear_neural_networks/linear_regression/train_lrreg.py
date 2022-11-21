"""train linear regression model in this directory"""

import torch
from synthetic_dataset import synthetic_data
from batch_dataset import iter_dataset
# from utils_train import squared_loss as squared_loss
from utils_train import * 
import numpy as np
import matplotlib.pyplot as plt

def train(features, labels, w, b, batch=16, lr=0.001, epochs = 20):
    """train linear regression model on a synthetic dataset
    
    args:
    features: torch.tensor
    labels: torch.tensor
    w: torch.tenosr
    b: torch.tensor
    batch: int
    lr: float
    epochs: int
    """
    for epoch in range(epochs):
        for X, y in iter_dataset(features, labels, batch):
            l = squared_loss(y, model(X, w, b))
            l.sum().backward(retain_graph=True)
            sgd([w, b], lr, batch)
        with torch.no_grad():
            train_loss = squared_loss(labels, model(features, w, b))
            print(f'epoch {epoch + 1}, loss {float(train_loss.mean()):f}')




if __name__ == "__main__":
    batch_size = 16
    epochs = 100
    lr = 1e-3
    w, b = initialize_params()
    features, labels = synthetic_data(w, b, 1000)
    print("Training....")
    train(features, labels, w, b, batch_size, lr, epochs)

    print('done!!')

