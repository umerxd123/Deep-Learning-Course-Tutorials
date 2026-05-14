# Tutorial 14_A — Simple RNN for Named Entity Recognition

## Overview

This tutorial focuses on building a Simple RNN model for Named Entity Recognition. The original tutorial used TensorFlow/Keras, but the implementation was completed in PyTorch.

The purpose of this tutorial was to understand how an RNN can process a sentence word by word and predict an entity label for each word.

## Objectives

The main objectives of this tutorial were:

- Understand the architecture of a Simple RNN
- Prepare a small NER dataset
- Tokenize sentences
- Convert words and labels into integer values
- Pad sequences to the same length
- Train an RNN for word-level sequence labeling
- Test the model on new sentences
- Change hidden units, epochs, and learning rate

## Dataset

A small custom Named Entity Recognition dataset was created inside the notebook.

The dataset contained sentences with four label types:

- PERSON
- LOCATION
- ORGANIZATION
- O

The label `O` was used for words that are not named entities.

Example:

```text
Elon Musk works at Tesla
