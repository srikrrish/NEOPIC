

import torch
import numpy as np

from torch.utils.data import TensorDataset
from _Utilities.utilities import normalize


def getDataloaders (configs):
    """
    Loads the required data for the Landau damping experiment and returns the train and test dataloaders.

    Returns:
        train_loader (torch.utils.data.DataLoader): train dataloader
        test_loader (torch.utils.data.DataLoader): test dataloader
    """
    path_X = configs['datapath']+'pos.npy'
    path_Q = configs['datapath']+'Eout.npy'

    input_x = np.load(path_X, allow_pickle=True)
    shape = input_x.shape
    p = np.random.permutation(shape[0])
    #inputs = torch.tensor(input_x, dtype=torch.float)
    inputs = torch.tensor(input_x[p,:], dtype=torch.float)
    input_q = np.load(path_Q, allow_pickle=True)
    #outputs = torch.tensor(input_q, dtype=torch.float)
    outputs = torch.tensor(input_q[p,:], dtype=torch.float)

    #inputs = normalize(inputs)
    #outputs = normalize(outputs)

    sub_sample_train = 2
    sub_sample_test = 2
    train_in = inputs[:configs['num_train'],::sub_sample_train]
    train_out = outputs[:configs['num_train'],::sub_sample_train]
    test_in = inputs[-configs['num_test']:,::sub_sample_test]
    test_out = outputs[-configs['num_test']:,::sub_sample_test]


    train_loader = torch.utils.data.DataLoader(TensorDataset(train_in, train_out), batch_size=configs['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(TensorDataset(test_in, test_out), batch_size=1, shuffle=False)


    return train_loader, test_loader

