# Multi Purpose Neural Network
## About

This program allows the user to create a neural network of any size and train it.
The number of layers and neurons is configurable, as is the learning rate and the number of epochs.
The user can choose the activation function for each layer. If the function has not been implemented yet, it can easily be done by computing the function and its derivative.
The input data for training and testing has to fit the Dataset class. Hence only numerical values.
If it is impossible, some minor changes in the Dataset and NeuralNetwork code should allow the new data to be processed.

## Usage
The program needs three arguments: the training file path, the testing file path, and the number of output neurons/solution columns.
The files should be in the CSV format, and the output neuron results should be placed at the beginning on the left.
The results are better if the data are normalized before usage. x: (x - x.min()) / (x.max() - x.min())).

## Iris Data set
About this Dataset
The Iris dataset was used in R.A. Fisher's classic 1936 paper, The Use of Multiple Measurements in Taxonomic Problems, and can also be found on the UCI Machine Learning Repository.
It includes three iris species with 50 samples each as well as some properties about each flower. One flower species is linearly separable from the other two, but the other two are not linearly separable from each other.
Hence the need for a multiperceptron, with at least one hidden layer.