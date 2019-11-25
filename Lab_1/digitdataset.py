import torch
import numpy as np
from torch.utils import data
from torch.utils.data import Dataset, DataLoader

class DigitDataset(Dataset):
    """ Digits Dataset """

    def __init__(self, datatype, device):
        assert datatype == 'train' or datatype == 'test'
        self.datatype = datatype
        xy = np.loadtxt('./pr_lab1_2016-17_data/' + datatype + '.txt')
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, 1:]).to(device).float()
        self.y_data = torch.from_numpy(xy[:, 0]).to(device).long()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def DigitDatasetLoader(N, frac, device):

    train_val_set = DigitDataset('train', device)
    n_train_val = len(train_val_set)
    n_train = int(frac*n_train_val)
    n_val = n_train_val - n_train
    train_set, val_set = data.random_split(train_val_set, (n_train, n_val))
    test_set = DigitDataset('test', device)
    n_test = len(test_set)

    train_loader = DataLoader(train_set, batch_size=N, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=N, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=N, shuffle=True)

    return train_loader, n_train, val_loader, n_val, test_loader, n_test
