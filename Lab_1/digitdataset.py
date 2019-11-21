import torch
import numpy as np
from torch.utils.data import Dataset

class DigitDataset(Dataset):
    """ Digits Dataset """


    def __init__(self, datatype, device):
        assert datatype == 'train' or datatype == 'test'
        self.datatype = datatype
        xy = np.loadtxt('./pr_lab1_2016-17_data/' + datatype + '.txt')
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, 1:]).to(device)
        self.y_data = torch.from_numpy(xy[:, 0]).to(device)


    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]


    def __len__(self):
        return self.len
