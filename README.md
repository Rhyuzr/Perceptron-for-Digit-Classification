# Perceptron-for-Digit-Classification


This project implements a Perceptron algorithm from scratch and compares it with scikit-learn's implementation.

## Project Structure

- [task1.py](cci:7://file:///c:/Users/youri/AppData/Local/Temp/0e9ad68b-cf8d-4cc9-a189-ea6e6171e217_GUALINO_HALMAERT%20%284%29.zip.217/MALIS/task1.py:0:0-0:0): Contains the main implementation of the Perceptron algorithm and comparison code

## Features

- Custom Perceptron implementation with:
  - Train method for learning from data
  - Predict method for making predictions
  - Adjustable learning rate (alpha)
- Comparison with scikit-learn's Perceptron implementation
- Example usage with AND logic gate dataset

## Requirements

- Python 3.x
- numpy
- scikit-learn

## Usage

The code demonstrates the implementation by:
1. Creating a simple AND logic gate dataset
2. Training both custom and scikit-learn Perceptrons
3. Comparing the predictions and learned parameters

The output will show:
- Predictions from both implementations
- Learned weights and biases from both implementations

## Implementation Details

The custom Perceptron implementation includes:
- Proper initialization of weights and bias
- Gradient descent-based weight updates
- Sign function for classification
- Error correction during training

## Comparison

The project includes a comparison between the custom implementation and scikit-learn's Perceptron, showing that both implementations can learn the same decision boundary for simple linearly separable problems like the AND gate.

## License

This project is for educational purposes only.
