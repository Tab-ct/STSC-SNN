import numpy as np
import torch
from torch.utils.data import Dataset

class SHD(Dataset):
    def __init__(self, train:bool, dt:int, T :int):
        super(SHD, self).__init__()
        
        # dt = 60ms and T = 15
        assert dt == 60, 'only SHD with dt=60ms is supported'
        self.train = train
        self.dt = dt
        self.T = T
        if train:
            X = np.load('./datasets/SHD/trainX_60ms.npy')[:,:T,:]
            y = np.load('./datasets/SHD/trainY_60ms.npy')
        else:
            X = np.load('./datasets/SHD/testX_60ms.npy')[:,:T,:]
            y = np.load('./datasets/SHD/testY_60ms.npy')

        self.len = 8156
        if train == False:
            self.len = 2264
        self.eventflow = X
        self.label = y
        
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.eventflow[idx, ...].astype(np.float32)    
        y = self.label[idx].astype(np.float32)                
        return (x, y)