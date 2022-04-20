# -*- coding: utf-8 -*-
"""
ResNet architecture and training/testing code for MLvCancer project

Assumes the training data is placed in same directory and has already been scaled
"""

import torch
import sklearn.metrics as metrics
import numpy as np
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
import torch.nn as nn
import matplotlib.pyplot as plt

# Custom DataSet class 
# --- Wraps TensorData to allow user to perform image transforms on data before training if desired
class TransformTensorDataset(Dataset):
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

# Residual block for ResNet
# -- consists of two convolutional layers with 2-d batch normalization between followed by a dropout layer and 
# --    ReLU activation function
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.6)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        # saving weights at beginning to reapply at end of block 
        identity = x

        # main block architecture - 2 conv layers with batch norm and relu
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        # dropout layer to help with overfitting
        x = self.dropout(x)

        # if there is a size mismatch, downsample identity before readding
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        # reapply weights from beginning of block
        x += identity
        x = self.relu(x)
        return x


# Defines whole Network based on residual blocks
#    -- Input layer w/ convolution, batch norm, and dropout
#    -- 3 Residual blocks as described above
#    -- Fully connected output layer
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        block=Block 
        image_channels=1
        num_classes=5
        layers = [2, 2, 2]
        self.in_channels = 64

        # input layer
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=5, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(0.75)

        # ResNetLayers
        self.layer1 = self.make_layers(block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self.make_layers(block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self.make_layers(block, layers[2], intermediate_channels=256, stride=2)

        # output layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
      # input layer fwding
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)

      # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

      # output layer
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    # function to make resnet layers based on given parameters
    def make_layers(self, block, num_residual_blocks, intermediate_channels, stride):
        layers = []

        # used if size of identity is incompatible with output of block
        identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels, kernel_size=1, stride=stride),
                                            nn.BatchNorm2d(intermediate_channels))
        layers.append(block(self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels # 256
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels)) # 256 -> 64, 64*4 (256) again
        return nn.Sequential(*layers)

# Use cuda device if possible
#   - Note: Although CPU can also be used if Cuda is unavailable, it will likely take an unreasonable amount of time to train
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

net = Network()
model = net.to(device)

# Initialize trainable parameters and hyperparameters
optimizer = optim.Adam(net.parameters(), lr = 0.01)
loss_func = nn.CrossEntropyLoss()
batch_size = 300
epochs = 150

# Loads scaled training data in chunks
data = np.load("X_train_scaled_00.npy")
for i in range(1, 10):
  data = np.concatenate((data, np.load("X_train_scaled_0{}.npy".format(str(i)))), axis=0)
for i in range(10, 20):
  data = np.concatenate((data, np.load("X_train_scaled_{}.npy".format(str(i)))), axis=0)
y = np.load("y_train.npy")

# Creates training and validation datasets
full_dataset = TransformTensorDataset([torch.tensor(data), torch.tensor(y)])
torch.manual_seed(10)
X_train, X_test = random_split(full_dataset, [686, 172])
train_dataloader = DataLoader(X_train, shuffle=True, batch_size=batch_size)
test_dataloader = DataLoader(X_test, shuffle=True, batch_size=batch_size)


print("Start training...")
for epoch in range(1,epochs+1):
    total_loss = 0
    total_images = 0
    total_correct = 0
    for batch in train_dataloader:           # Load batch
        images, labels = batch
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        images = images.to(device, dtype=torch.float)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)

        preds = model(images)             # Process batch

        loss = loss_func(preds, labels) # Calculate loss

        optimizer.zero_grad()
        loss.backward()                 # Calculate gradients
        optimizer.step()                # Update weights

        output = preds.argmax(dim=1)

        total_loss += loss.item()
        total_images += labels.size(0)
        total_correct += output.eq(labels).sum().item()

    model_accuracy = total_correct / total_images * 100

    print('ep {0}, loss: {1:.2f}, {2} train {3:.2f}%'.format(
            epoch, total_loss, total_images, model_accuracy), end='')


    print()

    if epoch % 2 == 0:
        torch.save(net.state_dict(),'checkModel{}.pth'.format(epoch))
        print("   Model saved to checkModel.pth")

    if model_accuracy > 99:
      break
    sys.stdout.flush()

torch.save(net.state_dict(),'savedModel.pth')
print("   Model saved to savedModel.pth")

# Function to test the model 
def test(path): 
    # Load model from given path
    model = Network()
    print('testing with path: ', path)
    model.load_state_dict(torch.load(path)) 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")    
    model.to(device) 
    running_accuracy = 0 
    total = 0 

    with torch.no_grad(): 
        for data in test_dataloader: 
            images, outputs = data 
            outputs = outputs.to(device, dtype=torch.float32) 
            images = images.to(device, dtype=torch.float)
            predicted_outputs = model(images) 
            _, predicted = torch.max(predicted_outputs, 1) 
            total += outputs.size(0) 
            running_accuracy += (predicted == outputs).sum().item() 
        print('Accuracy of the model based on the test set of the inputs is: %d %%' % (100 * running_accuracy / total))   
        return (100 * running_accuracy / total)
 

accuracies = []
epochs = []
for i in range(2, 152,2):
  epochs.append(i)
  accuracies.append(test('checkModel{}.pth'.format(str(i))))
plt.plot(epochs, accuracies)
plt.title('Testing Acc vs. Epochs of Training')
plt.savefig('testingacc.png')

mostAccurate = accuracies.index(max(accuracies))
print('Most accurate model is: checkModel{}.pth'.format(str(mostAccurate * 2)))
print('With accuracy: {} %', max(accuracies))
