import numpy as np
from sklearn.preprocessing import OneHotEncoder

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

class Perceptron(object):
    def __init__(self):
        """
        Initialize instance of OneHotEncoder for use in onehot function.
        """
        
        self.weights = None     # No need to change. Values for self.weights will be set in the fit function.
        self.classes = None     # No need to change. Values for self.classes will be set in the fit function.
        self.oh = OneHotEncoder()          # TODO Initialize OneHotEncoder

    def onehot(self, Y):
        """
        Helper function to encode the labels into one hot encoding format used in the one-vs-all classifier.
        Replace the class label from 0 and 1 to -1 and 1.

        Args:
                Y: list of class labels

        Return:
                onehotencoded: (N, C) numpy array where:
                                N is the number of datapoints in the list 'Y'
                                C is the number of distinct labels/classes in 'Y'

        Hint:
        1. It may be helpful to refer to sklearn documentation for the OneHotEncoder
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
        2. After the normal onehot encoding, replace the 0's with -1's because we will use this for the one-vs-all classifier

        """
        labels_reshaped = np.array(Y).reshape(-1,1)

        onehotencoded_sparse = self.oh.fit_transform(labels_reshaped)
        
        onehotencoded = onehotencoded_sparse.toarray()
        
        onehotencoded[onehotencoded == 0] = -1

        return onehotencoded

    def perceptron(self, X, Y, epochs=10):
        """
        1 vs all perceptron algorithm. We will use the regularization term (alpha) as 1.

        Args:
                X: (N, D+1) numpy array of the TF-IDF features for the data.
                Y: (N,) numpy vector of the one-hot encoded labels.
                epochs: Number of epochs for perceptron.

        Return:
                weight: (D+1, 1) weight matrix 

        Hint:
        1. Initialize weight to be 0s
        2. Read the documentation + code of the fit( ) method to understand what needs to be returned by this method.
        3. Note that a bias term has already been added to the dataset in the fit function.
        """
        N, D_plus_1 = X.shape
        weight = np.zeros((D_plus_1, 1))
        alpha = 1

        for epoch in range(epochs):
            for i in range(N):
                xi = X[i].reshape(-1,1)
                yi = Y[i]

                prediction = np.sign(np.dot(weight.T, xi))[0,0]

                if prediction != yi:
                    weight += alpha * yi * xi
        return weight

    def fit(self, data, labels):
        """
        Fit function for calculating the weights using perceptron.
        NOTE : This function is given and does not have to be implemented or changed.

        Args:
                data: (N, D) TF-IDF features for the data.
                labels: (N, ) list of class labels
        """

        bias_ones = np.ones((len(data), 1))
        X = np.hstack((data, bias_ones))
        Y = self.onehot(labels)

        self.classes = Y.shape[1]
        self.weights = np.zeros((X.shape[1], Y.shape[1]))

        for i in range(Y.shape[1]):
            W = self.perceptron(X, Y[:, i])
            self.weights[:, i] = W[:, 0]

    def predict(self, data):
        """
        Predict function for predicting the class labels.

        Args:
                data: (N, D) TF-IDF features for the data.

        Return:
                predictedLabels: (N,) numpy vector of predicted classes for the data.

        Hint: Remember to address the bias term. It may be helpful to see how the bias term was added in the fit function.
        """
        bias_ones = np.ones((data.shape[0], 1))
        X = np.hstack((data, bias_ones))

        scores = np.dot(X, self.weights)

        predicitons = np.argmax(scores, axis = 1)

        predictedlabels = self.oh.categories_[0][predicitons]

        return predictedlabels