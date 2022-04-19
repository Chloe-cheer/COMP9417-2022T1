"""Main program to train, evaluate and test mode."""

import os
import torch
import numpy as np
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, random_split
from algorithm.dataset import TransformTensorDataset, load_X
from algorithm.file_compressor import pre_processing

from algorithm.modules.cnn import Network
from algorithm.test import test_and_generate_acc_figure
from algorithm.train import train

"""
Original notebook is located at
    https://colab.research.google.com/drive/1SBfvNizBpxdYqOpQjhNgueqIp8kBAWdV

"""

def main():
    # Simple configs
    run_pre_processing = True
    split_ratio = "8:0:2" # train:val:test
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Hyper-parameters
    batch_size = 1
    num_epochs = 40

    # Pre-processing
    if run_pre_processing:
        pre_processing()

    # Paths
    base_path = "../input/"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    x_train_path = base_path+"X_train_chunks/"
    if not os.path.exists(x_train_path):
        os.makedirs(x_train_path)
    y_train_path = base_path+"y_train.npy"

    # Split sizes calcualtions
    split_ratio_lst = list(map(lambda x: int(x), split_ratio.split(":")))
    split_ratio_lst_total = sum(split_ratio_lst)
    data_size = 858
    split_sizes = [int(data_size*(r/split_ratio_lst_total)) for r in split_ratio_lst]
    split_sizes[-1] += data_size - sum(split_sizes)
    print(f"split_sizes are {split_sizes}")

    # Load data
    X = load_X(x_train_path)
    y = np.load(y_train_path)
    print("X.shape", X.shape)

    # Create data loaders
    full_dataset = TransformTensorDataset([torch.tensor(X), torch.tensor(y)])
    validation = False
    
    if len(split_sizes) == 3 and split_sizes[1] != 0:
        data_train, data_val, data_test = random_split(full_dataset, split_sizes)
        validation = True
    else:
        if len(split_sizes) == 3:
            split_sizes.remove(0)
        data_train, data_test = random_split(full_dataset, split_sizes)

    if validation:
        val_dataloader = DataLoader(data_val, shuffle=False, batch_size=batch_size)
    else:
        validation = True
        data_train, data_val = random_split(full_dataset, split_sizes)
        train_dataloader = DataLoader(data_train, shuffle=True, batch_size=batch_size)
        val_dataloader = DataLoader(data_val, shuffle=False, batch_size=batch_size)

    test_dataloader = DataLoader(data_test, shuffle=False, batch_size=batch_size)

    # Instantiate a network instance
    net = Network()
    model = net.to(device)

    # Print out model structure
    print(model)

    # Initilise optimizer and loss function
    optimizer = optim.Adam(net.parameters(), lr = 0.01)
    loss_func = nn.CrossEntropyLoss()

    # Train the model
    train(
        mdoel=model,
        device=device,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        loss_func=loss_func,
        validation=validation,
        val_dataloader=val_dataloader if validation else None,
    )

    # Test and generate an accuracy figure
    test_and_generate_acc_figure(test_dataloader=test_dataloader, num_epoch=num_epochs)

if __name__ == "__main__":
    main()