"""divide the dataset into mini-batches"""

import torch
import numpy as np
import random
from synthetic_dataset import synthetic_data

def iter_dataset(features, targets, batch_size = 16):
    """divide the dataset into mini-batches for training.
    
    Args
    features: torch.tensor (num_samples, num_features)
    targets: torch.tensor (num_samples, 1)
    batch_size: int
    
    Return
    mini-batch: torch.tensor((batch_size, num_feturse), (batch_size, 1))
    """
    num_samples = features.shape[0]
    indices = list(range(num_samples))
    random.shuffle(indices)

    for i in range(0, num_samples, batch_size):
        batch_indices = torch.tensor(indices[i:  min(i + batch_size, num_samples)])
        yield features[batch_indices], targets[batch_indices]

if __name__ == "__main__":
    true_w = torch.tensor([2, -3.4])
    true_b = torch.tensor([3.4])
    n = 4

    features, targets = synthetic_data(true_w, true_b, 1000)
    batch_sz = 10

    for X, y in iter_dataset(features, targets, batch_size= batch_sz):
        print(X, '\n', y)
        break
    print('the end!!')
