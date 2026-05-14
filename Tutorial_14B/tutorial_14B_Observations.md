# Tutorial 14_B — English to Urdu Translation using RNN

## Overview

This tutorial focuses on building an RNN-based English to Urdu translation model. The original tutorial used TensorFlow/Keras, but the implementation was completed in PyTorch.

The purpose of this tutorial was to understand how an encoder-decoder RNN can be used for sequence-to-sequence translation. The tutorial also included experiments with hidden units, epochs, and learning rate, along with a one-to-many RNN model for baby name generation.

## Objectives

The main objectives of this tutorial were:

- Understand the architecture of RNN
- Prepare an English–Urdu sentence-pair dataset
- Tokenize English and Urdu sentences
- Build an encoder-decoder RNN model
- Train the model for translation
- Generate Urdu translations for English sentences
- Change hidden units, epochs, and learning rate
- Develop a one-to-many RNN model for baby name generation

## Dataset

A small custom English–Urdu dataset was created inside the notebook.

Each data sample contained:

- English sentence
- Corresponding Urdu sentence

Example:

```text
English: how are you
Urdu: آپ کیسے ہیں
