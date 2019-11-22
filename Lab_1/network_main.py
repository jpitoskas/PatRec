import torch
import numpy as np
from torch.utils import data
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from digitdataset import DigitDataset
from twolayernn import TwoLayerNet

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"
if torch.cuda.is_available():
    print(  'DEVICE:', DEVICE,
            '\ncuda Version:', torch.version.cuda,
            '\nn_cuda =', torch.cuda.device_count(),
            '\nCuda Name(s):', torch.cuda.get_device_name(0)    )

train_val_set = DigitDataset('train', DEVICE)
n_train_val = len(train_val_set)
# 90% Train - 10 % Validation
n_train = int(0.9*n_train_val)
n_val = n_train_val - n_train
train_set, val_set = data.random_split(train_val_set, (n_train, n_val))

test_set = DigitDataset('test', DEVICE)
n_test = len(test_set)

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 128, 256, 100, 10

train_loader = DataLoader(train_set, batch_size=N, shuffle=True)
val_loader = DataLoader(val_set, batch_size=N, shuffle=True)
test_loader = DataLoader(test_set, batch_size=N, shuffle=True)

# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out)
# model.to(DEVICE)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

#function to calculate accuracy of model
def accuracy(y_hat, y):
     pred = torch.argmax(y_hat, dim=1)
     return (pred == y).float().sum()

def train_and_val_model(epochs, criterion, optimizer, model, train, validation, n_val):

    # TRAINING

    print("TRAINING...")

    for epoch in range(epochs+1):
        model.train()
        for data in train:
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(inputs)

            # Compute and print loss
            loss = criterion(y_pred, labels)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # VALIDATING

        model.eval()
        acc = 0
        with torch.no_grad():
            for data in validation:
                inputs, labels = data
                y_pred = model(inputs)
                acc += accuracy(y_pred, labels)
        if (epoch % 100 == 0):
            print("Epoch:", epoch, "with loss:", loss.item(), "and accuracy on validation:", acc.item()/n_val)
    
    torch.save(model.state_dict(), "./train.model")


def test_model(model, test, n_test):

    # TESTING

    print("TESTING...")

    model.eval()
    acc = 0
    with torch.no_grad():
        for data in test:
            inputs, labels = data
            y_pred = model(inputs)
            acc += accuracy(y_pred, labels)
    print("Test set accuracy:", acc.item()/n_test)

# Train and Validate model
train_and_val_model(100, criterion, optimizer, model, train_loader, val_loader, n_val)

# Test model
test_model(model, test_loader, n_test)