import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simple neural network model for classifying MNIST digits
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()

        self.flatten = nn.Flatten()  # Flatten the input image
        self.hidden_layer = nn.Linear(28 * 28, 64)  # First layer with 784 inputs and 64 outputs
        self.output_layer = nn.Linear(64, 10)  # Output layer with 10 classes (digits 0-9)

    def forward(self, x):
        x = self.flatten(x)  # Flatten the input
        x = F.relu(self.hidden_layer(x))  # Apply ReLU activation to the hidden layer
        x = self.output_layer(x)  # Output layer (no activation, will be handled by loss function)
        return x