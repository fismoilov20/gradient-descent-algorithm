# Gradient Descent: The Ultimate Optimizer

## Abstract

The process of adjusting hyperparameters, such as the step size, for an optimizer used in gradient-based machine learning algorithms can be a time-consuming and tedious task. Previous research has demonstrated that it is possible to optimize the step size along with the model parameters by manually deriving expressions for "hypergradients" in advance. 
In this article, we propose a modification to backpropagation that enables the *automatic* computation of hypergradients in a straightforward and elegant manner. This method can be applied to various optimizers and hyperparameters such as momentum coefficients, with ease. Furthermore, we can recursively apply the method to their own *hyper*-hyperparameters, resulting in an infinite cascade of optimizers. As the number of optimizers increases, they become less sensitive to the initial choice of hyperparameters. We conducted experiments on Multi-Layer Perceptron (MLP). Eventually, we provide a simple PyTorch implementation of this algorithm (https://github.com/fismoilov20/gradient-descent-algorithm.git).

*This repository contains an implementation of the algorithm in our paper.*

## Install

```bash
# install pytorch

# Windows, Linux, Mac
pip install torch torchvision torchaudio


# install gradient-descent-the-ultimate-optimizer
pip install gradient-descent-the-ultimate-optimizer
```

## Example

1. Build the MLP and initialize data loaders.

```python
import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class MNIST_FullyConnected(nn.Module):
    # A fully-connected neural network model designed for the MNIST dataset. This model is not an optimizer
    # itself but can be optimized using different optimization algorithms.
    def __init__(self, num_inp, num_hid, num_out):
        super(MNIST_FullyConnected, self).__init__()
        self.layer1 = nn.Linear(num_inp, num_hid)
        self.layer2 = nn.Linear(num_hid, num_out)

    def initialize(self):
        nn.init.kaiming_uniform_(self.layer1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.layer2.weight, a=math.sqrt(5))

    def forward(self, x):
        # Compute a prediction using the given input data.
        x = self.layer1(x)
        x = torch.tanh(x)
        x = self.layer2(x)
        x = torch.tanh(x)
        x = F.log_softmax(x, dim=1)
        return x

BATCH_SIZE = 256
EPOCHS = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
dl_train = torch.utils.data.DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True)
dl_test = torch.utils.data.DataLoader(mnist_test, batch_size=10000, shuffle=False)

model = MNIST_FullyConnected(28 * 28, 128, 10).to(DEVICE)
```

2. Next, import the `gradient_descent` package and create a stack of hyperoptimizers. In this example, we initialize a stack called `Adam/SGD`.

```python
from gradient_descent_algorithm import gda

optim = gdtuo.Adam(optimizer=gdtuo.SGD(1e-5))
```

`gdtuo.ModuleWrapper` enables the optimization of any `nn.Module using` hyperoptimizers.

```python
mw = gdtuo.ModuleWrapper(model, optimizer=optim)
mw.initialize()
```

Finally, utilize `mw` as an alternative to a PyTorch optimizer for optimizing the model. The training loop closely resembles the typical implementation in PyTorch, with any variations indicated by comments.

```python
for i in range(1, EPOCHS+1):
    running_loss = 0.0
    for j, (features_, labels_) in enumerate(dl_train):
        mw.begin() # call this before each step, enables gradient tracking on desired params
        features, labels = torch.reshape(features_, (-1, 28 * 28)).to(DEVICE), labels_.to(DEVICE)
        pred = mw.forward(features)
        loss = F.nll_loss(pred, labels)
        mw.zero_grad()
        loss.backward(create_graph=True) # important! use create_graph=True
        mw.step()
        running_loss += loss.item() * features_.size(0)
    train_loss = running_loss / len(dl_train.dataset)
    print("EPOCH: {}, TRAIN LOSS: {}".format(i, train_loss))
```

Note that on the first step of the train loop PyTorch will return the following warning:

```text
UserWarning: Using backward() with create_graph=True will create a reference cycle between the parameter and its gradient which can cause a memory leak. We recommend using autograd.grad when creating the graph to avoid this. If you have to use this function, make sure to reset the .grad fields of your parameters to None after use to break the cycle and avoid the leak.
```

This is normal and to be expected.
