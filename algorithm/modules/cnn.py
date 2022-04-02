"""Module containing code for a CNN object detector."""
import torch
import torchvision.models as models
from torch.nn import Linear, Module, Sigmoid


class CNN(Module):
    """A CNN model."""

    def __init__(self, classes, model) -> None:
        """Initialise the CNN."""
        super().__init__()
        # TODO
        ...

    def forward(self):
        """Data propagation."""
        # TODO
        ...