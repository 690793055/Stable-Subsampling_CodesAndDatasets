import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
import random

def calculate_acc(U, Z):
    """
    Calculates the Azadkia-Chatterjee Coefficient (ACC).

    Args:
        U (numpy.ndarray): A 1D numpy array representing samples of a random variable U.
        Z (numpy.ndarray): A 2D numpy array representing samples of a random vector Z, where each row is a sample.

    Returns:
        float: The calculated ACC value.
    """
    n = len(U)
    if n != Z.shape[0]:
        raise ValueError("The number of samples in U and Z must be the same.")

    R = np.zeros(n)
    L = np.zeros(n)
    M = np.zeros(n, dtype=int)  # Stores the index of the nearest neighbor

    # Calculate R_i and L_i
    for i in range(n):
        R[i] = np.sum(U <= U[i])  # Number of elements in U less than or equal to U[i]
        L[i] = np.sum(U >= U[i])  # Number of elements in U greater than or equal to U[i]

    # Build KDTree for efficient nearest neighbor search
    tree = KDTree(Z)

    # Find the nearest neighbor M(i) using KDTree
    for i in range(n):
        _, indices = tree.query(Z[i], k=2)  # k=2 to get the nearest neighbor excluding itself
        nearest_neighbor_index = indices[1] if indices[0] == i else indices[0]
        M[i] = nearest_neighbor_index

    # Calculate the numerator of ACC
    numerator = 0
    for i in range(n):
        numerator += (n * min(R[i], R[M[i]]) - L[i]**2)

    # Calculate the denominator of ACC
    denominator = 0
    for i in range(n):
        denominator += L[i] * (n - L[i])

    # Avoid division by zero
    if denominator == 0:
        return 0.0  # Or return other value or raise exception depending on the specific situation

    # Calculate ACC
    acc = numerator / denominator
    return acc

def GSL_calculate(matrix):
    """
    Iterates through each column of a matrix, calculates the ACC of that column with the rest of the matrix,
    and returns the mean of all ACC values.

    Args:
        matrix (numpy.ndarray): The input numpy matrix.

    Returns:
        float: The mean of all calculated ACC values. Returns 0 if the matrix has only one column.
    """
    num_cols = matrix.shape[1]
    if num_cols < 2:
        print("Warning: Input matrix has only one column, cannot calculate ACC of column with the rest, returning 0.")
        return 0.0

    acc_values = []
    num_rows = matrix.shape[0]

    for i in range(num_cols):
        # Use the current column as U
        U = matrix[:, i]

        # Use the remaining columns as Z
        remaining_cols_indices = [j for j in range(num_cols) if j != i]
        Z = matrix[:, remaining_cols_indices]

        # Calculate the ACC of the current column and the rest
        acc = calculate_acc(U, Z)
        acc_values.append(acc)

    # Calculate the mean of all ACC values
    gls = abs(np.mean(acc_values))
    return gls