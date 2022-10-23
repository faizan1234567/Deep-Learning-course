"""synthetic dataset generation for simple linear regression model"""

import math
import torch
import numpy as np

def synthetic_data(w, b, num_samples):
    """generate synthetic for linear regression model design and training
    y = Xw + b + noise
    
    Args
    w: float (a number)
    b: float (bais)
    num_samples: int (number of examples in the dataset)
    
    return
    dataset: torch.tensor (num_samples, len(w))
    """
    X = torch.normal(0, 1, (num_samples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

if __name__ == "__main__":
    true_w = torch.tensor([2, -3.4])
    true_b = torch.tensor([3.4])
    features, targets = synthetic_data(true_w, true_b, 1000)
    print("features shape: {}, labels shape: {}".format(features.shape, targets.shape))

    


