"""Main program to train, evaluate and test mode."""

import argparse
import os
import torch
import sys

def main(config):
    """Run a thorough flow of the main program."""
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    if not cuda:
        print('Using CPU')
    else:
        print(f'{device}: {torch.cuda.get_device_name(device)}')
    
    # TODO

def load(config):
    """Load the given dataset from file."""
    # TODO
    pass

def save(config, model, loss):
    """Save a model locally as a file."""
    # TODO
    pass

def train(config, model, loss, train_data, val_data, metrics):
    """Model training."""
    # TODO
    pass

def evaluate(config, model, data, metrics):
    """Model evaluation."""
    # TODO
    pass

def test(config, model, data):
    """Model testing."""
    # TODO
    pass

def parse_args(args):
    """Parse the system arguments from the command line.
    
    Returns:
    return a config object that stores of config parameters.
    
    Reference:
    https://docs.python.org/3/library/argparse.html#module-argparse
    """
    parser = argparse.ArgumentParser()
    configs_group = parser.add_argument_group('Configs')
    params_group = parser.add_argument_group('Model Paramaters')

    configs_group.add_argument(
        '--input-dir',
        default=os.path.relpath(os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            'input'
        )),
        type=str,
        required=False,
        help='The path of which the dataset is located in.'
    )
    configs_group.add_argument(
        '--output-dir',
        default=os.path.relpath(os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            'output'
        )),
        type=str,
        required=False,
        help='The path to save the predictions.'
    )
    configs_group.add_argument(
        '--log-dir',
        default=os.path.relpath(os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            'logs'
        )),
        type=str,
        required=False,
        help='The path to store log files.'
    )
    params_group.add_argument(
        '--epochs',
        default=2,
        type=int,
        required=False,
        help='The number of epochs for model training.'
    )
    # TODO: Add more command line argument options that suit your needs

    return parser.parse_args(args)

if __name__ == '__main__':
    args = sys.argv[1:]
    
    main(parse_args(args))
    
