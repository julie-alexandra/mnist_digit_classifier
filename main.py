import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import DigitClassifier  # Import the model defined in model.py



# Define a transform to convert images to tensor
image_to_tensor = transforms.ToTensor()

# Load the MNIST dataset
train_data = datasets.MNIST(
   root='data', # Directory to store the dataset
   train=True, # Use the training set
   download=True, # Download the dataset if not already present
   transform=image_to_tensor # Apply the tensor transform
)

test_data = datasets.MNIST(
   root='data', 
   train=False, 
   download=True, 
   transform=image_to_tensor
)

# Create data loaders to handle data in batches (= more efficient)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


def show_sample_images():
    """Function to display a few sample images from the training dataset."""

    # Create an iterator over the training dataloader
    training_data_iterator = iter(train_loader) # an iterator is an object that gives the next item in a sequence
   
    # Get the first batch from the iterator
    images, labels = next(training_data_iterator) # data iter is returning 2 items: images and labels

    # Set up a grid to display 6 images from the batch 
    plt.figure(figsize=(8, 6))
    for i in range(6):
        plt.subplot(2, 3, i + 1) #2 rows, 3 columns
        plt.imshow(images[i].squeeze(), cmap='gray') # squeeze to remove single-dimensional entries
        plt.title(f'Label: {labels[i].item()}') # item() to get the Python number from a tensor
        plt.axis('off') # Hide axes

    plt.tight_layout() # Adjust layout to prevent overlap
    plt.show() # Display the images 

# Function to train the model
def train_model(model, train_loader, loss_function, optimizer, epochs):
    for epoch in range(epochs): # Loop over the number of epochs (complete passes through the dataset)
        total_loss = 0

        for images, labels in train_loader:
            # Forward pass
            outputs = model(images)
            loss = loss_function(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad() # Zero the gradients before backward pass 
            loss.backward() # Backward pass to compute gradients
            optimizer.step() # Update the weights of the model using the gradients

            total_loss += loss.item() # convert loss to a Python number and accumulate it

        avg_loss = total_loss / len(train_loader) # Average loss for the epoch
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}') # Print the average loss for the epoch


def evaluate_model(model, test_loader):
    """Function to evaluate the model on the test dataset."""
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient computation for efficiency
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy


def save_model(model, path):
    """Save the trained model."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path):
    """Load a trained model."""
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model


def test_predictions(model, test_loader, num_samples=5):
    """Display predictions on a few test samples."""
    model.eval()
    test_iter = iter(test_loader)
    images, labels = next(test_iter)
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    plt.figure(figsize=(12, 8))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f'True: {labels[i].item()}\nPred: {predicted[i].item()}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    print("Displaying sample images from the MNIST dataset:")
    show_sample_images() # Call the function to display images

    model = DigitClassifier() # Create an instance of the model
    loss_function = nn.CrossEntropyLoss() # Define the loss function for classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Define the optimizer

    print("Starting training of the model...")
    train_model(model, train_loader, loss_function, optimizer, epochs=10)
    
    print("\nEvaluating the model...")
    evaluate_model(model, test_loader)
    
    print("\nSaving the model...")
    save_model(model, 'digit_classifier.pth')
    
    print("\nTesting predictions on sample images:")
    test_predictions(model, test_loader, num_samples=5)

else:
    print("This module is intended to be run as a script to display sample images.")    

