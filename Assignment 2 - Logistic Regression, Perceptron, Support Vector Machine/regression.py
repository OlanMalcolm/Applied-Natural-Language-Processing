import pandas as pd
import numpy as np
from scipy.special import softmax
from sklearn.preprocessing import OneHotEncoder

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


class Regression(object):
    def __init__(self):
        """
        Initialize instance of OneHotEncoder in self.oh for use in onehot function.
        """
        
        self.W = None           # No need to change. Values for self.W will be set in the fit function
        self.oh = OneHotEncoder()         # TODO Initialize OneHotEncoder

    def onehot(self, labels):
        """
        Helper function to encode a labels into one hot encoding format

        Args:
                labels: list of class labels

        Return:
                onehotencoded: (N, C) numpy array where:
                        N is the number of datapoints in the list 'labels'
                        C is the number of distinct labels/classes in 'labels'

        Hints: 
            1. .toarray() may be helpful in converting a sparse matrix into a numpy array
            2. Recall that fit_transform takes in a 2d array input instead of a 1d array input
        """
        labels_reshaped = np.array(labels).reshape(-1,1)
        onehotencoded = self.oh.fit_transform(labels_reshaped).toarray()
        return onehotencoded

    def gradient(self, X, Y, W):
        """
        Apply softmax function to compute the predicted labels and calculate the gradients of the loss w.r.t the weights weights.

        Args:
                X: (N, D) numpy array of the TF-IDF features for the data.
                Y: (N, C) numpy array of the one-hot encoded labels.
                W: (D, C) numpy array of weights.

        Return:
                gradient: (D,C) numpy array of the computed gradients

        Hint: Use the formula in Section 1.1 of HW2.ipynb to compute the gradients
        """
        mu = 0.01  # Do not change mu value.
        N = X.shape[0]

        raw_scores = X @ W

        probs = softmax(raw_scores, axis = 1)

        gradient = ((1/N) * (X.T @ (probs - Y))) + (2 * mu * W)

        return gradient

    def gradient_descent(self, X, Y, epochs=10, eta=0.1):
        """
        Basic gradient descent algorithm with fixed eta and mu

        Args:
                X: (N, D) numpy array of the TF-IDF features for the data.
                Y: (N, C) numpy array of the one-hot encoded labels.
                epochs: Number of epochs for the gradient descent (optional, defaults to 10).
                    Note that epoch refers to one pass through the full training dataset
                eta: Learning rate (optional, defaults to 0.1)

        Return:
                weight: (D,C) weight matrix

        Hint: Weight should be initialized to be zeros
        """
        D = X.shape[1]
        C = Y.shape[1]

        weight = np.zeros((D,C))

        for epoch in range(epochs):
            grad = self.gradient(X, Y, weight)
            weight -= eta * grad

        return weight

    def fit(self, data, labels):
        """
        Fit function for calculating the weights using gradient descent algorithm.
        NOTE : This function is given and does not have to be implemented or changed.

        Args:
                data: (N, D) TF-IDF features for the data.
                labels: (N, ) list of class labels
        """

        X = np.asarray(data)
        Y_onehot = self.onehot(labels)
        self.W = self.gradient_descent(X, Y_onehot)

    def predict(self, data):
        """
        Predict function for predicting the class labels. 
        It may be helpful to refer back to the Jupyter Notebook to review
        the workflow for computing logistic regression.

        Args:
                data: (N, D) TF-IDF features for the data.

        Return:
                predictedLabels: (N,) 1D array of predicted classes for the data.

        Hint: You can assume that fit will be called before predict and that self.W is available for use in this function
        """
        X = np.asarray(data)
        raw_scores = X @ self.W
        probs = softmax(raw_scores, axis = 1)
        predictions = np.argmax(probs, axis = 1)
        return predictions
