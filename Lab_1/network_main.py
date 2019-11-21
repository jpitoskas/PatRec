import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from digitdataset import DigitDataset
from twolayernn import TwoLayerNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(  'DEVICE:', DEVICE,
#         '\ncuda Version:', torch.version.cuda,
#         '\nn_cuda =', torch.cuda.device_count(),
#         '\nCuda Name(s):', torch.cuda.get_device_name()    )
# print(torch.cuda.device(0))

train_set = DigitDataset('train')
test_set = DigitDataset('test')

train_loader = DataLoader(train_set, batch_size=100, shuffle=True, num_workers=4)

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 100, 256, 100, 1

# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out).double()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

epochs = 5
for epoch in range(epochs):
    for idx, data in enumerate(train_loader):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(inputs)

        # Compute and print loss
        loss = criterion(y_pred, labels)
        print(epoch, idx, loss.data)

#       Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
