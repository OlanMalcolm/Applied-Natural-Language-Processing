import numpy as np
from sklearn.preprocessing import OneHotEncoder
from concurrent.futures import ProcessPoolExecutor  # You may not need to use this
from functools import partial # You may not need to use this

class OHE_BOW(object): 
    def __init__(self):
        '''
        Initialize instance of OneHotEncoder in self.oh for use in fit and transform
        If needed, you may set handle_unknown to ignore when initalizing OneHotEncoder.
        '''
        self.vocab_size = None          # keep
        self.oh = OneHotEncoder()                 # initialize

    def split_text(self, data):
        '''
        Helper function to separate each string into a list of individual words

        Args:
            data: list of N strings
        
        Return:
            data_split: list of N lists of individual words from each string

        '''
        data_split = []
        for sent in data:
            words = sent.split()
            data_split.append(words)
    
        return data_split

    def flatten_list(self, data):
        '''
        Helper function to flatten a list of list of words into a single list

        Args:
            data: list of N lists of W_i words 
        
        Return:
            data_split: (W,) numpy array of words, 
                where W is the sum of the number of W_i words in each of the list of words      

        '''
        data_split = [item for sublist in data for item in sublist]

        return data_split


    def fit(self, data):
        '''
        Fit the initialized instance of OneHotEncoder to the given data
        Use split_text to separate the given strings into a list of words and 
        flatten_list to flatten the list of words in a sentence into a single list of words
        Reference the documentation for OneHotEncoder for the data shape it is expecting

        Set self.vocab_size to the number of unique words in the given data corpus

        Args:
            data: list of N strings 
        
        Return:
            None

        Hint: You may find numpy's reshape function helpful when fitting the encoder

        '''
        text = self.split_text(data)
        flat_text = self.flatten_list(text)
        flat_text_array = np.array(flat_text).reshape(-1,1)
        self.oh.fit(flat_text_array)
        self.vocab_size = len(self.oh.categories_[0])

    def onehot(self, words):
        '''
        Helper function to encode a list of words into one hot encoding format

        Args:
            words: list of W_i words from a string
        
        Return:
            onehotencoded: (W_i, D) numpy array where:
                W_i is the number of words in the current input list i
                D is the vocab size

        Hint:   .toarray() may be helpful in converting a sparse matrix into a numpy array
                You can use sklearn's built-in OneHotEncoder transform function
        '''
        array = np.array(words).reshape(-1,1)
        hot_transform = self.oh.transform(array)
        onehotencoded = hot_transform.toarray()

        return onehotencoded

    def transform(self, data):
        '''
        Use the already fitted instance of OneHotEncoder to help you transform the given 
        data into a bag of words representation. You will need to separate each string 
        into a list of words and then encode this list of words into the one-hot format. 
        Using the one-hot encoding of each sentence, convert it to the bag-of-words representation.
        
        For any empty strings append a (1, D) array of zeros instead.
            
        
        Args:
            data: list of N strings
        
        Return:
            bow: (N, D) numpy array

        Hints: 
            1. Using a try and except block during one hot encoding transform may be helpful 
            2. If using ProcessPoolExecutor, you may create a helper function inside of the OHE_BOW class
        '''
        bow_list = []
        for sentence in data:
            words = sentence.split()
            if len(words) == 0:
                # empty string → zero vector of vocab size
                bow_vector = np.zeros(self.vocab_size)
            else:
                try:
                    onehot_encoded = self.onehot(words)    # shape (W_i, D)
                    bow_vector = onehot_encoded.sum(axis=0)  # sum to get bag-of-words vector (D,)
                except Exception as e:
                    # If transform fails (maybe unseen word), return zero vector
                    bow_vector = np.zeros(self.vocab_size)
            bow_list.append(bow_vector)
        
        bow = np.vstack(bow_list)   # shape (N, D)
        return bow