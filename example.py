"""
In this example, we will train a neural network to approximate
the sigmoid function f, given by
    
    f(x) = 1/(1+exp(-x)).
"""


import numpy as np

import amlgrad.neurallib.nn as nn
from amlgrad.neurallib.tensor import Tensor
from amlgrad.neurallib.optim import SGD
from amlgrad.neurallib.loss import MSE


# Data
X = np.random.uniform(-5, 5, (10000,1))
y = 1/(1+np.exp(-1*X))

X, y = Tensor(X), Tensor(y)

# Neural network
class MyNet(nn.NeuralNet):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(1, 20),
                       nn.ReLU(),
                       nn.Linear(20, 40),
                       nn.ReLU(),
                       nn.Linear(40,1)]

# Initialize model, loss and optimizer
model = MyNet()
criterion = MSE()
optim = SGD(model.parameters(), lr=0.0001)

# Traning loop
n_epoch = 400
for epoch in range(n_epoch):
    # Calculate the predictions of the model
    preds = model.forward(X)

    # Calculate the loss of the predictions
    loss = criterion.forward(preds, y)

    # Calculate the gradients of the loss wrt the parameters.
    loss.backward()

    # Optimize the parameters
    optim.step()

    # Zero the gradients
    loss.zero_grad()

    print(f"Epoch:{epoch+1}, Loss:{loss.data}")