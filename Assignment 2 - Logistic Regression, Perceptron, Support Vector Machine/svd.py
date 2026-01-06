import numpy as np


class SVD(object):
    def __init__(self):
        pass

    def svd(self, data):
        """
        Do SVD. You could use numpy SVD. 
        Args:
                data: (N, D) TF-IDF features for the data.

        Return:
                U: (N,N) numpy array
                S: (min(N,D), ) numpy array
                V^T: (D,D) numpy array
        """
        U, S, Vt = np.linalg.svd(data, full_matrices = True)
        return U, S, Vt

    def rebuild_svd(self, U, S, V, k):
        """
        Rebuild SVD by k componments.

        Args:
                U: (N,N) numpy array
                S: (min(N,D), ) numpy array
                V: (D,D) numpy array
                k: int corresponding to number of components

        Return:
                data_rebuild: (N,D) numpy array

        Hint: numpy.matmul may be helpful for reconstruction.
        """
        U_k = U[:, :k]
        S_k = np.diag(S[:k])
        V_k = V[:k, :]

        data_rebuild = np.matmul(np.matmul(U_k, S_k), V_k)
        
        return data_rebuild

    def compression_ratio(self, data, k): 
        """
        Compute the compression ratio: (num stored values in compressed)/(num stored values in original)

        Args:
                data: (N, D) TF-IDF features for the data.
                k: int corresponding to number of components

        Return:
                compression_ratio: float of proportion of storage used
        """
        N, D = data.shape

        original_storage = N * D

        compressed_storage = (N*k) + k + (k * D)

        compressed_ratio = compressed_storage / original_storage

        return compressed_ratio

