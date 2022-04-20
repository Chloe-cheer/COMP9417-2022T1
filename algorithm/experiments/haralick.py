"""
Experiments for Haralick / Logistic Regression

"""


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import mahotas as mt
import gc 

from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics

# np.load() implementation
y_train = np.load('y_train.npy')
X_train = np.load('X_train.npy', mmap_mode='r')

# pytorch load implementation
from torch.utils.data import DataLoader

train_data = []
test_data = []
i = 0 

# First 600 rows are our "train data"
while i < 600:
    train_data.append([np.array(X_train[i]),y_train[i]])
    i += 1
    
# Rest are "test data"
while i < len(X_train):
    test_data.append([np.array(X_train[i]),y_train[i]])
    i = i + 1
    
train_dataloader = DataLoader( train_data, batch_size = 20)
test_dataloader = DataLoader( test_data, batch_size = 20)

del train_data
del test_data
gc.collect()

# Display image and label
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap = 'gray')
plt.show()
print(f"Label: {label}")

def extract_features(image):
    # calculate haralick texture features for 4 types of adjacency
    textures = mt.features.haralick(image)

    # take the mean of it and return it
    ht_mean = textures.mean(axis=0)
    return ht_mean

simplified_training_data = []
i = 0 
for batch_features, batch_label in iter(train_dataloader):
    for batch_feature in batch_features:
        gray =  np.dot(batch_feature[...,:3], [76.24499999999999, 149.685, 29.07]) #combines unormalisation and 
        haralick_feature = extract_features(gray.astype(int))
        simplified_training_data.append(haralick_feature)
        print(str(int(100*i/600)) + "% done.")
        i = i + 1 

i = 0
simplified_test_data = []
for batch_features, batch_label in iter(test_dataloader):
    for batch_feature in batch_features:
        gray =  np.dot(batch_feature[...,:3], [76.24499999999999, 149.685, 29.07]) #combines unormalisation and 
        haralick_feature = extract_features(gray.astype(int))
        simplified_test_data.append(haralick_feature)
        print(str(int(100*i/257)) + "% done.")
        i += 1

"""
Simple logistic regression experiments
"""

lm = linear_model.LogisticRegression(max_iter = 10000)
lm.fit(simplified_training_data, y_train[0:600])
lm.score(simplified_training_data, y_train[0:600])
lm.score(simplified_test_data, y_train[600:])
