## writing some helpers functions to get model trained on the synthetic dataset
'''
1. initialize models params to random
2. write model
3. write loss function
4. calculate gradients
5. update model parameters'''

import torch

def initialize_params(w_shape = (2, 1), b_shape= (1,1)):
    """initialize model params to random
    
    args:
    w_shape: tuple
    b_shape: tuple
    
    return
    w: torch.tensor
    b: torch.tensor"""
    w = torch.normal(0, 0.01, size = list(w_shape), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return w, b


def model(X, w, b):
    """define linear regression model f(x) = wX + b
    
    Args
    X: torch.tensor -> batch of data samples
    w: torch.tensor -> parameter w
    b: torch.tensor -> offset or bais
    
    return 
    y_hat: torch.tensor -> prediction of the model using w,b
    """
    return torch.matmul(X, w) + b

def squared_loss(y, y_hat):
        
    """squared loss using y and prediction

    Args
    y: torch.tensor -> true values
    y_hat: torch.tensor -> predicted values

    return
    loss value"""

    loss = (y - y_hat.reshape(y.shape))**2/2
    return loss

def sgd(params, lr, batch_size): #@save
    """Minibatch stochastic gradient descent."""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

if __name__ == "__main__":
    print("params initialization...")
    w, b = initialize_params()
    print("w: {}".format(w))
    print("b: {}".format(b))



