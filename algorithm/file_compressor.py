import torch
from torch import nn
import numpy as np
import torch
import torchvision
import numpy as np
from numpy import save

def pre_processing():
    """Pre-process X_train data from X_train_chunks."""

    for i in range(0, 20):
        X_train = np.load("../input/X_train_chunks/X_train_{}.npy".format(i))
        print("X_train.shape", X_train.shape)

        X_train = np.transpose(X_train, (0,3,1,2))
        X_train_0 = torchvision.transforms.functional.rgb_to_grayscale(torch.tensor(X_train))

        X_train_scaled_0 = nn.functional.interpolate(torch.tensor(X_train_0), scale_factor=0.25)
        print("X_train_scaled_0.shape", X_train_scaled_0.shape)

        path = "../input/X_train_chunks/X_train_scaled_{:02d}.npy".format(i)
        save(path, X_train_scaled_0)
        print(f"Saved {path}")