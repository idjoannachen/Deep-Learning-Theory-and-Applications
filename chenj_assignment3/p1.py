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
grad_w1 = 2 * X.T @ X @ w_init - 2 * X.T @ y
grad_w1

# b) calculate gradient with respect to the weights w using autograd # first create the activation function
# print(w_init)
if w_init.grad:
  w_init.grad.data.zero_()
loss = torch.pow((X @ w_init - y), 2).sum()
loss.backward()
print(w_init.grad)

# c) check that the two are equal
print(w_init.grad == grad_w1)
print(torch.abs(w_init.grad - grad_w1) / torch.abs(grad_w1))

# The reason why it returns false is precision. We computed their relative difference and found the difference 
# is very small (<10e-7). Therefore, we conclude that the gradient is the same as we expected.
