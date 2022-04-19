"""Testing utilities."""

import datetime
import os
import time
import torch
from algorithm.modules.cnn import Network
import matplotlib.pyplot as plt

def test(path, test_dataloader): 
    """Function to test the mode."""
    model = Network()
    print('testing with path: ', path)
    model.load_state_dict(torch.load(path)) 
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
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

    end = time.time()
    print(f"\nTotal testing time: {str(datetime.timedelta(seconds=end-start))}")