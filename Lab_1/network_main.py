import torch
import numpy as np
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

train_set = DigitDataset('train', DEVICE)
test_set = DigitDataset('test', DEVICE)

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 100, 256, 100, 10

train_loader = DataLoader(train_set, batch_size=N, shuffle=True)
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

# TRAINING

print("TRAINING...")

epochs = 1000
for epoch in range(epochs):
    model.train()
    for idx, data in enumerate(train_loader):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(inputs)

        # Compute and print loss
        # print(y_pred.shape, labels.shape)
        loss = criterion(y_pred, labels)
        print(epoch, idx, loss.data[0])

#       Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # print("Epoch:", epoch, "with loss:", loss.item(), "and accuracy:", acc.item()/len(test_set))

    # TESTING

    # print("TESTING...")
    model.eval()
    acc = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            y_pred = model(inputs)
            # print(y_pred)
            # _, predicted = torch.max(y_pred.data, 1)
            # print(predicted)
            # print(labels)
            # correct += (predicted.long() == labels.long()).sum().item()
            acc += accuracy(y_pred, labels)
    if (epoch % 10 == 0):
        # print('Accuracy of the model:', acc.item()/len(test_set))
        print("Epoch:", epoch, "with loss:", loss.item(), "and accuracy:", acc.item()/len(test_set))
