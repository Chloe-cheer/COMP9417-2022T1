"""Utilities for loading and preprocessing the given dataset."""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import mahotas as mt
import gc
import skimage.measure
from torch.utils.data import DataLoader, random_split
import torch

def get_dataloaders():
  y_train = np.load('y_train.npy')
  X_train = np.load('X_train.npy', mmap_mode='r')
 
  n_train = 686
  n_validate = 172

  full_data = [] 
  i = 0 

  ## First 686 rows are our "train data"
  print("Preparing training data...")
  while i < n_train:
      full_data.append([np.array(X_train[i]),y_train[i]])
      i = i + 1

  print("Training data finished.")
  print("Preparing validation data...")
  ## Last 172 are "validation data"
  while i < len(X_train):
      full_data.append([np.array(X_train[i]),y_train[i]])
      i = i + 1

  print("Validation data finished.")

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

def extract_features(image):
    # calculate haralick texture features for 4 types of adjacency
    textures = mt.features.haralick(image)

    # take the mean of it and return it
    ht_mean = textures.mean(axis=0)
    return ht_mean
  
def get_features(input_dataloader):
    haralick_features = []
    pooling_features = []
    y = []
    for batch_features, batch_label in iter(input_dataloader):
        j = 0 
        while j < len(batch_features):
            gray =  np.dot(batch_features[j][...,:3], [76.245, 149.685, 29.07]) #combines unormalisation and 
            haralick_feature = extract_features(gray.astype(int))
            haralick_features.append(haralick_feature)
            gray_normalised =  np.dot(batch_features[j][...,:3], [0.299, 0.587, 0.114])
            averaged = skimage.measure.block_reduce(gray_normalised, (64,64), np.average)
            flat_array = averaged.flatten()
            pooling_features.append(flat_array)
            y.append(batch_label[j])
            j = j + 1 
    return haralick_features, pooling_features, y
