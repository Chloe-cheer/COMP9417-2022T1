"""Utilities for loading and preprocessing the given dataset."""
import numpy as np
import pandas as pd
import mahotas as mt
import gc
import skimage.measure
from torch.utils.data import Dataset, DataLoader, random_split
import torch

class TransformTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

def load_X(x_train_path):
    X = None
    for i in range(20):
        if X is None:
            X = np.load(x_train_path+f"X_train_scaled_{i:02d}.npy")
        else:
            X = np.concatenate((X, np.load(x_train_path+f"X_train_scaled_{i:02d}.npy")), axis=0)
    return X

def get_dataloaders():
  y_train = np.load('y_train.npy')
  X_train = np.load('X_train.npy', mmap_mode='r')
 
  full_data = [] 
  i = 0 

  print("Loading data...")
  while i < len(X_train):
      full_data.append([np.array(X_train[i]),y_train[i]])
      i = i + 1

  print("Packaging data into DataLoaders...")
 
  torch.manual_seed(10)
  train_data, validate_data = random_split(full_data, [686, 172])
  train_dataloader = DataLoader(train_data, shuffle=True)
  validate_dataloader = DataLoader(validate_data, shuffle=True)


  print("Freeing up memory...")
  del full_data
  del train_data
  del validate_data
  gc.collect()
  
  return train_dataloader, validate_dataloader

def extract_haralick(image):
    # We compute the mean of the 4 thirteen dimensional vectors formed from the 4 types of adjacency GLCM matrices and return it.
    all_textures = mt.features.haralick(image)
    avg_haralick_texture = all_textures.mean(axis=0)
    return avg_haralick_texture
  
def get_features(input_dataloader):
    haralick_features = []
    pooling_features = []
    y = []
    for batch_features, batch_label in iter(input_dataloader):
        j = 0 
        while j < len(batch_features):
            gray =  np.dot(batch_features[j][...,:3], [76.245, 149.685, 29.07]) #combines unormalisation and grey-scale conversion
            haralick_feature = extract_haralick(gray.astype(int))
            haralick_features.append(haralick_feature)
            gray_normalised =  np.dot(batch_features[j][...,:3], [0.299, 0.587, 0.114])
            averaged = skimage.measure.block_reduce(gray_normalised, (64,64), np.average)
            flat_array = averaged.flatten()
            pooling_features.append(flat_array)
            y.append(batch_label[j])
            j = j + 1 
    return haralick_features, pooling_features, y
