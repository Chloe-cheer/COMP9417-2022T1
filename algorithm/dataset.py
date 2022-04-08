"""Utilities for loading and preprocessing the given dataset."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset

class DatasetLoader(Dataset):
    def __init__(self):
        pass

    def __getitem__(self):
        pass

    def pre_processing(self):
        pass