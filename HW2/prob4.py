""" Neural Network.
A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with PyTorch. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).
"""

import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # activation functions
import torch.optim as optim  # optimizer
from torch.autograd import Variable # add gradients to tensors
from torch.nn import Parameter # model parameter functionality

import torchvision
import torchvision.datasets as datasets

import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)

# ##################################
# import data
# ##################################
# download the MNIST dataset
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)


# separate into data and labels
# training data
# reducing training dataset to 1000 points and test dataset to 2000 points in order to
# create an overfitting model on which to study regularization

train_data = mnist_trainset.data.to(dtype=torch.float32)[:1000]
train_data = train_data.reshape(-1, 784)
train_labels = mnist_trainset.targets.to(dtype=torch.long)[:1000]

print("train data shape: {}".format(train_data.size()))
print("train label shape: {}".format(train_labels.size()))

# testing data
test_data = mnist_testset.data.to(dtype=torch.float32)[:2000]
test_data = test_data.reshape(-1, 784)
test_labels = mnist_testset.targets.to(dtype=torch.long)[:2000]

print("test data shape: {}".format(test_data.size()))
print("test label shape: {}".format(test_labels.size()))

# load into torch datasets
train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)

# ##################################
# set hyperparameters
# ##################################
# Parameters
learning_rate = 0.01
num_epochs = 1000 # training for a long time to see overfitting
batch_size = 128

# Network Parameters
n_hidden_1 = 128  # 1st layer number of neurons
n_hidden_2 = 128  # 2nd layer number of neurons
n_hidden_3 = 128  # 3rd layer number of neurons
num_input = 784  # MNIST data input (img shape: 28*28)
num_classes = 10  # MNIST total classes (0-9 digits)

# ##################################
# defining the model 
# ##################################

# method 1
class FCNN(nn.Module):
    def __init__(self, input_dim, output_dim, p=None):
        super(FCNN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.p = p

        self.layer1 = nn.Linear(input_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.layer4 = nn.Linear(n_hidden_3, num_classes)

        self.nonlin1 = nn.Sigmoid()
        self.nonlin2 = nn.Sigmoid()
        self.nonlin3 = nn.Sigmoid()

    def forward(self, x):
        if self.p:
            h1 = nn.Dropout(p=self.p)(self.nonlin1(self.layer1(x)))
            h2 = nn.Dropout(p=self.p)(self.nonlin2(self.layer2(h1)))
            h3 = nn.Dropout(p=self.p)(self.nonlin3(self.layer3(h2)))
        else:
            h1 = self.nonlin1(self.layer1(x))
            h2 = self.nonlin2(self.layer2(h1))
            h3 = self.nonlin3(self.layer3(h2))
            
        output = self.layer4(h2)

        return output

# alternative way of defining a model in pytorch
# you can create an equivalent model to FCNN above
# using nn.Sequential
#
# model2 = nn.Sequential(nn.Linear(num_input, n_hidden_1),
#                        nn.Sigmoid(),
#                        nn.Linear(n_hidden_1, n_hidden_2),
#                        nn.Sigmoid(),
#                        nn.Linear(n_hidden_2, num_classes))

# ##################################
# helper functions
# ##################################
def get_accuracy(output, targets):
    """calculates accuracy from model output and targets
    """
    output = output.detach()
    predicted = output.argmax(-1)
    correct = (predicted == targets).sum().item()

    accuracy = correct / output.size(0) * 100

    return accuracy


def to_one_hot(y, c_dims=10):
    """converts a N-dimensional input to a NxC dimnensional one-hot encoding
    """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    c_dims = c_dims if c_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], c_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot

# ##################################
# creata dataloader
# ##################################
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True)


# ##################################
# main training function - example
# ##################################

def train():

    model = FCNN(num_input, num_classes)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # initialize loss list
    metrics = [[0, 0]]

    # iterate over epochs
    for ep in range(num_epochs):
        model.train()

        # iterate over batches
        for batch_indx, batch in enumerate(trainloader):

            # unpack batch
            data, labels = batch

            # get predictions from model
            pred = model(data)

            ########################
            # TO DO
            ########################
            # here is where the loss function would compare 
            # the model's output to the targets and caculate the loss 
            # for this batch. You may use PyTorch loss modules in 
            # torch.nn to train the model using
            # 1. MSE Loss
            # 2. Cross-Entropy Loss

            # loss = some_loss_function(pred, labels)

            # backpropagate the loss
            loss.backward()

            # update parameters
            optimizer.step()

            # reset gradients
            optimizer.zero_grad()

        # compute full train and test accuracies 
        # every epoch
        model.eval() # model will not calculate gradients for this pass
        train_ep_pred = model(train_data)
        test_ep_pred = model(test_data)

        train_accuracy = get_accuracy(train_ep_pred, train_labels)
        test_accuracy = get_accuracy(test_ep_pred, test_labels)

        # print loss every 100 epochs
        if ep % 100 == 0:
            print("train acc: {}\t test acc: {}\t at epoch: {}".format(train_accuracy,
                                                                 test_accuracy,
                                                                 ep))
        metrics.append([train_accuracy, test_accuracy])

    return np.array(metrics), model

# so using the training function, you would ultimately 
# be left with your metrics (in this case accuracy vs epoch) and 
# your trained model.
# Ex. 
# metric_array, trained_model = train()

def plot_accuracies_v_epoch(metric_array):
    pass

# HINT for 4.1 AND 4.2: 
# we can pass in arguments to our training function that
# may be hyperparameters, loss functions, regularization terms etc.