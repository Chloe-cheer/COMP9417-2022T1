import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from torch import nn
import torch
import torchvision
from numpy import save

from algorithm.modules.cnn import Network


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
    save_base_path = "../input"

    if not os.path.exists(save_base_path):
        os.makedirs(save_base_path)

    X_test = np.load(f"{save_base_path}/X_test.npy", mmap_mode='r')

    X_test = np.transpose(X_test, (0,3,1,2))
    X_test_0 = torchvision.transforms.functional.rgb_to_grayscale(torch.tensor(X_test))

    X_test_scaled_0 = nn.functional.interpolate(torch.tensor(X_test_0), scale_factor=0.25)
    print("X_test_scaled_0.shape", X_test_scaled_0.shape)

    save_path = f"{save_base_path}/X_test_scaled.npy"
    save(save_path, X_test_scaled_0)
    print(f"Saved {save_path}")

def genereate_preds_for_preds_submission():
    """Generate predictions from the given X_test"""
    input_base_path = "../input"
    output_base_path = "../output"

    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)

    data = np.load(f"{input_base_path}/X_test_scaled.npy") # Assume X_test has been preprocessed

    test_dataloader = DataLoader(data, shuffle=False)

    def test(path): 
        # Load the model that we saved at the end of the training loop 
        model = Network()
        print('testing with path: ', path)
        model.load_state_dict(torch.load(path)) 
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device")    
        model.to(device) 

        with torch.no_grad(): 
            predictions = []
            for data in test_dataloader: 
                images = data 
                images = images.to(device, dtype=torch.float)
                predicted_outputs = model(images) 
                _, predicted = torch.max(predicted_outputs, 1) 
                predictions.append(predicted.cpu().numpy())
            return predictions

    predictions = np.squeeze(np.asarray(test(f'{output_base_path}/best-model.pth')))
    print("predictions.shape", predictions.shape)
    print(predictions)
    np.save(f'{output_base_path}/testPredictionsNew.npy', predictions)