# Neural Network - MNIST Digit Classifier

## Description

This project implements a simple neural network to classify handwritten digits from the MNIST dataset. The model uses PyTorch to create, train, and evaluate a digit classifier (0-9).

## Objectives

- Understand the basics of neural networks
- Learn to use PyTorch for deep learning
- Implement an image classifier
- Evaluate model performance

## Project Architecture

```
neural-network/
├── main.py              # Main script with training and evaluation
├── model.py             # Neural network architecture definition
├── requirements.txt     # Python dependencies
├── README.md           # Project documentation
├── data/               # MNIST dataset (automatically downloaded)
│   └── MNIST/
└── nnenv/              # Python virtual environment
```

## Model Architecture

The `DigitClassifier` model is a simple neural network with:

- **Input layer**: 784 neurons (28×28 flattened pixels)
- **Hidden layer**: 64 neurons with ReLU activation
- **Output layer**: 10 neurons (one for each digit 0-9)

```python
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.hidden_layer = nn.Linear(28 * 28, 64)
        self.output_layer = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x
```

## Installation and Setup

### Prerequisites
- Python 3.8+
- pip

### 1. Clone the project
```bash
git clone <repository-url>
cd neural-network
```

### 2. Create and activate virtual environment
```bash
python -m venv nnenv
source nnenv/bin/activate  # On macOS/Linux
# or
nnenv\Scripts\activate     # On Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Running the main script
```bash
python main.py
```

The script will:
1. Display sample images from the MNIST dataset
2. Train the model for 5 epochs
3. Evaluate accuracy on the test set
4. Save the trained model
5. Test predictions on sample images

### Available features

#### 1. Data visualization
```python
show_sample_images()  # Display 6 example images
```

#### 2. Model training
```python
train_model(model, train_loader, loss_function, optimizer, epochs=5)
```

#### 3. Evaluation
```python
evaluate_model(model, test_loader)  # Calculate accuracy
```

#### 4. Save/Load
```python
save_model(model, 'digit_classifier.pth')
load_model(model, 'digit_classifier.pth')
```

#### 5. Prediction testing
```python
test_predictions(model, test_loader, num_samples=5)
```

## Expected Results

With default parameters, the model should achieve:
- **Accuracy**: ~97-98% on the test set
- **Training time**: 2-3 minutes on CPU
- **Final loss**: ~0.10 after 5 epochs

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 0.001 | Adam optimizer learning rate |
| Batch Size | 64 | Training batch size |
| Epochs | 5 | Number of complete passes through the dataset |
| Hidden Units | 64 | Number of neurons in the hidden layer |

## Possible Improvements

1. **Architecture**:
   - Add more hidden layers
   - Use convolutional neural networks (CNN)
   - Implement dropout to reduce overfitting

2. **Optimization**:
   - Test different optimizers (SGD, RMSprop)
   - Adjust learning rate
   - Implement learning rate scheduler

3. **Visualization**:
   - Plot loss and accuracy curves
   - Confusion matrix
   - Learned weights visualization

## Concepts Learned

- **Neural networks**: Architecture, forward pass, backpropagation
- **PyTorch**: Tensors, DataLoader, optimizers, loss functions
- **Classification**: Softmax, cross-entropy loss
- **Evaluation**: Accuracy, validation on test set
- **Best practices**: Virtual environments, model saving

## Troubleshooting

### Import error
```bash
ImportError: cannot import name 'DigitClassifier' from 'model'
```
**Solution**: Check that the `model.py` file is correct and the virtual environment is activated.

### Visualization issues
If matplotlib doesn't display, install the appropriate backend:
```bash
pip install matplotlib --upgrade
```

## License

This project is for educational purposes as part of BeCode training.


---

**Author**: [Your name]  
**Training**: BeCode - Data & AI Bootcamp  
**Date**: August 2025