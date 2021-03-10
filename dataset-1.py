import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import math


class WineDataset(Dataset):
    def __init__(self):
    #Data loading
        xy = np.loadtxt('./data/wine/wine.csv', delimiter =',', dtype=np.float32, skiprows=1 )
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]]) #n_samples, 1
        self.n_samples = xy.shape[0]
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    #datset [0]

    def __len__(self):
        return self.n_samples

    
Dataset = WineDataset()
first_data = Dataset[0]
features, labels = first_data
print(features, labels)