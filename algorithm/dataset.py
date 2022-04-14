"""Utilities for loading and preprocessing the given dataset."""
import numpy as np
import gc 
from torch.utils.data import DataLoader

def get_dataloaders():
  y_train = np.load('y_train.npy')
  X_train = np.load('X_train.npy', mmap_mode='r')

  # split the data in 60-20-20 
  n_train = 514
  n_validate = 172

  train_data = []
  validate_data = []
  test_data = []
  i = 0 

  ## First 514 rows are our "train data"
  print("Preparing training data...")
  while i < n_train:
      train_data.append([np.array(X_train[i]),y_train[i]])
      i = i + 1

  print("Training data finished.")
  print("Preparing validation data...")
  ## Last 172 are "test data"
  while i < n_train + n_validate:
      validate_data.append([np.array(X_train[i]),y_train[i]])
      i = i + 1

  print("Validation data finished.")
  print("Preparing test data...")
  ## Last 172 are "test data"
  while i < len(X_train):
      test_data.append([np.array(X_train[i]),y_train[i]])
      i = i + 1

  print("Test data finished.")
  print("Packaging data into DataLoaders...")
  train_dataloader = DataLoader( train_data, batch_size = 20)
  validate_dataloader = DataLoader( validate_data, batch_size = 20)
  test_dataloader = DataLoader( test_data, batch_size = 20)

  print("Freeing up memory...")
  del train_data
  del validate_data
  del test_data
  gc.collect()
  
  return train_dataloader, validate_dataloader, test_dataloader 
