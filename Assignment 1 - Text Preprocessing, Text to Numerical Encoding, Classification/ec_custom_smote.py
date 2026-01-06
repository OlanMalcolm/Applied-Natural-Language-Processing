import numpy as np
from sklearn.neighbors import NearestNeighbors
np.random.seed(42)

### SUBMIT THIS FILE TO THE BONUS EXTRA CREDIT GRADESCOPE SECTION ###
def custom_smote(samples, n, k):
    """
    Implement the SMOTE algorithm on the training data.
    For this function, you can use scikit-learn's NearestNeighbors function to find
    the k nearest neighbors.

    Hint:
        1. randomly pick a minority sample
        2. find its k nearest neighbors
        3. randomly pick one of the k nearest neighbors and generate a synthetic sample
        4. when k equals 1, the closest neighbor to the current point (not the point itself) is selected 
           as the nearest neighbor

    Args: 
        samples: (N, D) ndarray of minority class samples
        n: Number of synthetic samples to generate
        k: Number of nearest neighbors
    Return: 
        synthetic: (n, D) ndarray of synthetic samples
    """
    N, D = samples.shape
    synthetic = np.zeros((n, D))

    # Fit the nearest neighbors model
    nn = NearestNeighbors(n_neighbors=k+1)  # +1 because the point itself is included
    nn.fit(samples)

    for idx in range(n):
        # Step 1: Randomly choose an index of the minority class sample
        i = np.random.randint(0, N)
        sample = samples[i].reshape(1, -1)

        # Step 2: Find k nearest neighbors (excluding self)
        neighbors = nn.kneighbors(sample, return_distance=False).flatten()

        # Remove self (index 0), select one of the k neighbors
        neighbor_idx = np.random.choice(neighbors[1:])  # exclude the first (self)

        # Step 3: Interpolate between sample[i] and neighbor
        diff = samples[neighbor_idx] - samples[i]
        gap = np.random.rand()  # random weight between 0 and 1
        synthetic[idx] = samples[i] + gap * diff

    return synthetic
    

