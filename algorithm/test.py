"""Testing utilities."""

import datetime
import os
import time
import torch
from algorithm.modules.cnn import Network
import matplotlib.pyplot as plt

def test(path, test_dataloader): 
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
            images, outputs = data 
            outputs = outputs.to(device, dtype=torch.float32) 
            images = images.to(device, dtype=torch.float)
            predicted_outputs = model(images) 
            _, predicted = torch.max(predicted_outputs, 1) 
            predictions.append(predicted.cpu().numpy())
        
        return predictions

def test_and_generate_acc_figure(test_dataloader, num_epochs):
    start = time.time()

    accuracies = []
    epochs = []
    for i in range(2, num_epochs+1, 2):
        epochs.append(i)
        accuracies.append(test('checkModel{}.pth'.format(str(i)), test_dataloader))
    plt.plot(epochs, accuracies)
    plt.title('Testing Acc vs. Epochs of Training')

    stats_directory = "stats/"
    if not os.path.exists(stats_directory):
        os.makedirs(stats_directory)

    plt.savefig(f'{stats_directory}/testingacc.png')

    mostAccurate = accuracies.index(max(accuracies))
    print('Most accurate model is: checkModel{}.pth'.format(str(mostAccurate * 2)))
    print('With accuracy: {} %', max(accuracies))

    end = time.time()
    print(f"\nTotal testing time: {str(datetime.timedelta(seconds=end-start))}")
