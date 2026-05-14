# Tutorial 02 — MLP Classifier

## Overview

This tutorial focuses on training a Multi-Layer Perceptron classifier on the Iris dataset. The model was implemented in PyTorch instead of using the direct Scikit-learn MLPClassifier.

The main objectives were to train an MLP model, scale the input data, evaluate the model, visualize the learning curve, and observe how hidden layers, neurons, and learning rate affect the training process.

## Dataset

The Iris dataset was used for this tutorial. It contains 150 flower samples with four input features:

- Sepal length
- Sepal width
- Petal length
- Petal width

The model classifies the samples into three classes:

- Setosa
- Versicolor
- Virginica

## Data Scaling

Before training, the input features were scaled using StandardScaler. This is important because neural networks train better when all input features are on a similar scale.

Scaling helps the optimizer update the weights more smoothly and prevents one feature from dominating the training process.

## Baseline MLP Model

The baseline MLP model used two hidden layers with 10 neurons in each layer.

The structure was:

- Input layer: 4 features
- Hidden layer 1: 10 neurons
- ReLU activation
- Hidden layer 2: 10 neurons
- ReLU activation
- Output layer: 3 classes

CrossEntropyLoss was used because this is a multi-class classification problem.

## Baseline Learning Curve

![Baseline MLP Learning Curve](baseline_learning_curve.png)

The baseline loss curve shows that the training loss decreased steadily over the epochs. The loss dropped quickly in the early part of training and then continued decreasing more slowly.

This shows that the model was learning properly and gradually improving its predictions.

## Baseline Training Accuracy

![Baseline MLP Training Accuracy](baseline_training_accuracy.png)

The training accuracy increased as the number of epochs increased. At the beginning, the accuracy was low, but it improved quickly after the first few epochs.

Near the end of training, the model reached almost perfect training accuracy, showing that the MLP learned the Iris dataset very well.

## Task 1 — Changing Hidden Layers and Neurons

The tutorial task required modifying the MLP model to test different numbers of hidden layers and neurons. This was done to observe how model size affects accuracy and the learning curve.

The tested architectures were:

- [8]
- [16]
- [32]
- [16, 8]
- [32, 16]
- [64, 32]
- [64, 32, 16]
- [128, 64, 32]

## Hidden Layer Comparison Results

![Hidden Layer Comparison Results](hidden_layer_results_table.png)

From the results, all architectures performed reasonably well on the Iris dataset. The smaller models were able to learn the dataset, but larger models achieved lower final training loss.

The deeper and wider models such as [64, 32, 16] and [128, 64, 32] produced very low training loss and reached 100% training accuracy.

However, the test accuracy did not increase much beyond 93.33%. This shows that increasing the number of layers and neurons improves training performance, but it does not always improve test performance on a small dataset.

## Architecture Learning Curves

![Learning Curves for Different MLP Architectures](architecture_learning_curves.png)

The architecture comparison curve shows that larger networks converged faster and reached lower final loss values.

The smallest model, [8], had the slowest loss reduction. Larger models such as [64, 32] and [128, 64, 32] reduced the loss much faster.

This means that increasing model capacity can improve convergence speed. However, because the Iris dataset is small and simple, very large models are not necessary.

## Task 2 — Changing Learning Rate

The second task was to change the learning rate and observe its effect on convergence.

The tested learning rates were:

- 0.1
- 0.01
- 0.001
- 0.0001

## Learning Rate Comparison

![Learning Curves for Different Learning Rates](learning_rate_curves.png)

The learning rate comparison shows a clear difference in convergence behavior.

The learning rate 0.0001 decreased the loss very slowly and still had a relatively high loss after 300 epochs. This means the learning rate was too small for fast convergence.

The learning rate 0.001 was stable but slower than 0.01.

The learning rate 0.01 gave a smooth and fast decrease in loss, making it a good balance between speed and stability.

The learning rate 0.1 converged quickly, but the curve had sharp spikes, showing unstable behavior. This means the learning rate was too high and caused large weight updates.

## Best Configuration

Based on the results, larger architectures reached lower training loss, but the test accuracy stayed close for multiple models.

A good practical architecture for this dataset is [64, 32] or [64, 32, 16], because they achieved low training loss and good test accuracy without being unnecessarily large.

For learning rate, 0.01 gave the best balance because it converged quickly while remaining more stable than 0.1.

## Key Observations

- Scaling the data improved the training process.
- Increasing hidden layers and neurons reduced training loss.
- Larger models learned faster, but did not always improve test accuracy.
- Very small learning rates converged slowly.
- Very large learning rates caused unstable loss curves.
- A moderate learning rate gave the best balance between convergence speed and stability.

## Conclusion

This tutorial helped in understanding how an MLP classifier works in PyTorch. The experiments showed that model architecture and learning rate strongly affect the training behavior.

For the Iris dataset, a moderate-sized MLP with a suitable learning rate was enough to achieve good performance.
