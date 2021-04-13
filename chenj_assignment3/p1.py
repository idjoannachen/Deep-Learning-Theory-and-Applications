import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from sklearn.datasets import fetch_california_housing
# import sample data
housing = fetch_california_housing()
m,n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data]
X = torch.Tensor(housing['data'])
y = torch.Tensor(housing['target']).unsqueeze(1)
# create the weight vector
w_init = torch.randn(8,1,requires_grad=True)
# TO DO:
# a) calculate closed form gradient with respect to the weights

# b) calculate gradient with respect to the weights w using autograd # first create the activation function

# c) check that the two are equal
