# Tutorial 01 — Building a Perceptron

## Overview
This tutorial focuses on building a perceptron from scratch and understanding how weights and bias are updated during training. The task was originally explained using TensorFlow/basic Python style, and I implemented the same concept using PyTorch.

## Objective
The main objective was to classify whether a flower belongs to the Iris-setosa class or not using the Iris dataset.

## Dataset
The Iris dataset contains 150 samples with four features:

- Sepal length
- Sepal width
- Petal length
- Petal width

The original dataset has three classes, but for this tutorial it was converted into a binary classification problem:

- Iris-setosa
- Not Iris-setosa

## Classic Perceptron Implementation
In the first implementation, I used labels `1` and `-1`.

The perceptron was built from scratch using PyTorch tensors. The model calculates a weighted sum and applies a step function. If the output is different from the actual label, the weights and bias are updated manually.

## Weight Update Rule
The perceptron updates the weights using:

`W = W + learning_rate * (actual - predicted) * input`

The bias is also updated using the prediction error.

## Results
The model was trained and tested on the Iris dataset. The training errors decreased over epochs, showing that the perceptron was learning to separate Iris-setosa from the other two classes.

The final accuracy was shown in the notebook output.

## Manual Input Prediction
A manual input function was added where the user can enter the four flower features. The trained model then predicts whether the flower is Iris-setosa or not.

## Sigmoid Modification
As required in the task, the step function was replaced with a sigmoid activation function.

For this version, the labels were changed from `1` and `-1` to `1` and `0`.

The sigmoid model outputs a probability between 0 and 1. A threshold of 0.5 was used for final classification.

## Key Observations
- The classic perceptron uses hard classification with a step function.
- The sigmoid version gives a probability-based output.
- PyTorch requires a more manual training process compared to TensorFlow/Keras.
- Writing the training loop manually helped in understanding forward pass, loss calculation, backpropagation, and optimizer updates.

## Conclusion
This tutorial helped in understanding the basic structure of a perceptron and how the same idea can be implemented in PyTorch. Both the classic perceptron and sigmoid-based version were implemented successfully.
