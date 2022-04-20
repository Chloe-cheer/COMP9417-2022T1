import os
import torch
from torch import nn
import numpy as np
import torch
import torchvision
import numpy as np
from numpy import save

def pre_processing_X_train():
    """Pre-process X_train data from X_train_chunks."""
    save_base_path = "../input/X_train_chunks"

    if not os.path.exists(save_base_path):
        os.makedirs(save_base_path)

    for i in range(0, 20):
        X_train = np.load(f"{save_base_path}/X_train_{i}.npy")
        print("X_train.shape", X_train.shape)

        X_train = np.transpose(X_train, (0,3,1,2))
        X_train_0 = torchvision.transforms.functional.rgb_to_grayscale(torch.tensor(X_train))

        X_train_scaled_0 = nn.functional.interpolate(torch.tensor(X_train_0), scale_factor=0.25)
        print("X_train_scaled_0.shape", X_train_scaled_0.shape)

        path = f"{save_base_path}/X_train_scaled_{i:02d}.npy"
        save(path, X_train_scaled_0)
        print(f"Saved {path}")

def pre_processing_X_test():
    """Pre-process X_test data."""
    save_path = "X_test_scaled.npy"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    X_test = np.load("X_test.npy", mmap_mode='r')

    X_test = np.transpose(X_test, (0,3,1,2))
    X_test_0 = torchvision.transforms.functional.rgb_to_grayscale(torch.tensor(X_test))

    X_test_scaled_0 = nn.functional.interpolate(torch.tensor(X_test_0), scale_factor=0.25)
    print("X_test_scaled_0.shape", X_test_scaled_0.shape)

    save(save_path, X_test_scaled_0)
    print(f"Saved {save_path}")
