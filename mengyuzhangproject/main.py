import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn import Linear, ReLU, Sigmoid, Module, BCELoss
from torch.nn.init import kaiming_uniform_, xavier_uniform_
import pandas as pd
from utils.args import parse_args
from utils.load_data import process_data,load_data
import random



# Model parameter
args = parse_args()




# MLP Model
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 64)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(64, 128)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # third hidden layer and output
        self.hidden3 = Linear(128, 1)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        X = self.act3(X)
        return X








# random seed
def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)




def train(train_data_x, train_data_y, test_data_x, test_data_y):
    for epoch in range(args.epochs):
        train_loss = 0.0
        optimizer.zero_grad()
        output = model(train_data_x)
        loss = criterion(output.reshape(1000),train_data_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*train_data_x.size(0)
        train_loss = train_loss / len(train_data_y)
        print('Epoch:  {}  \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))
    test(test_data_x, test_data_y)



def test(data,target):
    correct = 0
    total = 0
    with torch.no_grad():
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum()
    print('Accuracy of the network on the test datasets: {:.6f}'.format(
        100 * correct / total))
    return 100.0 * correct / total



# model, optimizer，loss_function
model = MLP(args.n_inputs)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)


def main():
    for repetition in range(args.repetitions):
        set_random_seeds(repetition)
        # load data
        rawdata = (process_data()).astype(float)
        data_y = torch.tensor(rawdata[:, -1])
        data_x = torch.tensor(rawdata[:, 0:-1])
        train_data_x, train_data_y, test_data_x, test_data_y = load_data(data_x,data_y,args)
        train(train_data_x.to(torch.float32), train_data_y.to(torch.float32), test_data_x.to(torch.float32), test_data_y.to(torch.float32))

if __name__ == "__main__":
    main()

