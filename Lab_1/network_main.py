import torch
import numpy as np
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt
from digitdataset import DigitDataset, DigitDatasetLoader
from twolayernn import TwoLayerNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
if torch.cuda.is_available():
    print(  'DEVICE:', DEVICE,
            '\ncuda Version:', torch.version.cuda,
            '\nn_cuda =', torch.cuda.device_count(),
            '\nCuda Name(s):', torch.cuda.get_device_name(0)    )

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 128, 256, 100, 10

# 90% Train - 10 % Validation
train_loader, n_train, val_loader, n_val, test_loader, n_test = DigitDatasetLoader(N, 0.9, DEVICE)

# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out).to(DEVICE)

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

    if os.path.exists("./train.model"):
        model.load_state_dict(torch.load("./train.model"))
        # model.eval()
        print("Loaded from pre-trained model")
    
    print("TRAINING...")

    total_epochs = []
    total_acc = []
    total_loss = []
    for epoch in range(epochs+1):
        model.train()
        for data in train:
            # get the inputs
            inputs, labels = data

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

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
        if (epoch % 100 == 0):
            total_epochs.append(epoch)
            total_acc.append(acc.item()/n_val)
            total_loss.append(loss.item())

    # torch.save(model.state_dict(), "./train.model")

    plt.figure()
    plt.title("Train and Validation")
    plt.xlabel("Epochs")
    plt.plot(total_epochs, total_loss, 'o-', color="r",
             label="Training Loss")
    plt.plot(total_epochs, total_acc, 'o-', color="g",
             label="Validation Accuracy")
    plt.legend(loc="best")
    plt.grid()
    plt.show()


def test_model(model, test, n_test):

    # TESTING

    print("TESTING...")

    model.eval()
    acc = 0
    with torch.no_grad():
        for data in test:
            inputs, labels = data

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            y_pred = model(inputs)
            acc += accuracy(y_pred, labels)
    print("Test set accuracy:", acc.item()/n_test)

# Train and Validate model
train_and_val_model(100, criterion, optimizer, model, train_loader, val_loader, n_val)

# Test model
test_model(model, test_loader, n_test)
