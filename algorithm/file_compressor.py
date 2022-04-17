import torch

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

# for i in range(0, 20):

#     X_train = np.load("../input/X_train_chunks/X_train_{}.npy".format(i))
#     print(X_train.shape)
#     # print('before', np.unique(X_train))

#     X_train = np.transpose(X_train, (0,3,1,2))
#     X_train_0 = torchvision.transforms.functional.rgb_to_grayscale(torch.tensor(X_train))

#     X_train_scaled_0 = nn.functional.interpolate(torch.tensor(X_train_0), scale_factor=0.25)
#     print(X_train_scaled_0.shape)
#     # print('after', np.unique(X_train_scaled_0))

#     save("../input/X_train_chunks/X_train_scaled_{:02d}.npy".format(i), X_train_scaled_0)


X_test = np.load("X_test.npy", mmap_mode='r')

X_test = np.transpose(X_test, (0,3,1,2))
X_test_0 = torchvision.transforms.functional.rgb_to_grayscale(torch.tensor(X_test))

X_test_scaled_0 = nn.functional.interpolate(torch.tensor(X_test_0), scale_factor=0.25)
print(X_test_scaled_0.shape)
# print('after', np.unique(X_test_scaled_0))

save("X_test_scaled.npy", X_test_scaled_0)