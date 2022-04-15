import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import numpy as np
import torch
import torchvision
import sklearn.metrics as metrics
import numpy as np
import sys
import matplotlib.pyplot as plt
import statistics
from numpy import asarray
from numpy import save

for i in range(0, 10):

    X_train = np.load("../input/X_train_chunks/X_train_{}.npy".format(str(i)))
    print(X_train.shape)

    X_train = np.transpose(X_train, (0,3,1,2))
    X_train_0 = torchvision.transforms.functional.rgb_to_grayscale(torch.tensor(X_train))

    X_train_scaled_0 = nn.functional.interpolate(torch.tensor(X_train_0), scale_factor=0.25)
    print(X_train_scaled_0.shape)

    save("X_train_scaled_0{}.npy".format(str(i)), X_train_scaled_0)

for i in range(10, 20):

    X_train = np.load("X_train_chunks/X_train_{}.npy/X_train_{}.npy".format(str(i), str(i)))

    print(X_train.shape)

    X_train = np.transpose(X_train, (0,3,1,2))
    X_train_0 = torchvision.transforms.functional.rgb_to_grayscale(torch.tensor(X_train))

    X_train_scaled_0 = nn.functional.interpolate(torch.tensor(X_train_0), scale_factor=0.25)
    print(X_train_scaled_0.shape)

    save("X_train_scaled_{}.npy".format(str(i)), X_train_scaled_0)
