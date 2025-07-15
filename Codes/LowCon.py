"""
%%
% Project Name: LowCon
% Description: The LowCon algorithm
% Author: Yang Jinjing
% Email: yangjinjing94@163.com
% Date: 2025-04-19
%%

% INPUTS
%   m       : the full sample points
%   d       : the OLHD design points
%
%
% OUTPUT 
%   cm      : the global stability loss of the USSP subsample points
%   id      ：the points index of the USSP subsample points
"""
    
    

import numpy as np
from sklearn.neighbors import KDTree
from scipy.optimize import linear_sum_assignment


def normalization(data, new_min=-1, new_max=1):
    """Linearly normalizes data to the specified range"""
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    # Handle the case of constant columns (avoid division by zero)
    ranges = data_max - data_min
    ranges[ranges == 0] = 1
    return (data - data_min) / ranges * (new_max - new_min) + new_min

def LowCon(m, d):
    """
    Args:
        m (np.ndarray): Full sample point matrix (n_samples, n_features)
        d (np.ndarray): Design point matrix (n_design, n_features)

    Returns:
        np.ndarray: Array of nearest neighbor indices (n_design,)
    """

    # Data normalization
    m_norm = normalization(m, -1, 1)
    d_norm = normalization(d, -1, 1)

    # Create KDTree structure
    tree = KDTree(m_norm, metric='euclidean')

    # Perform nearest neighbor search
    distances, indices = tree.query(d_norm, k=1)

    # Flatten the results and convert to MATLAB-style indexing (starting from 1)
    return indices  # Remove +1 for zero-based indexing


def LowCon_without_rep(m, d):
    """
    Args:
        m (np.ndarray): Full sample point matrix, shape (n_samples, n_features)
        d (np.ndarray): Design point matrix, shape (n_design, n_features)

    Returns:
        np.ndarray: Array of nearest neighbor indices (n_design,), where each d is assigned to a unique m
    """
    # Data normalization (assuming normalization is already defined)
    m_norm = normalization(m, -1, 1)
    d_norm = normalization(d, -1, 1)

    # Convert data type to float32 to reduce memory consumption
    m_norm = m_norm.astype(np.float32)
    d_norm = d_norm.astype(np.float32)

    # Calculate the Euclidean distance matrix between d_norm and m_norm
    # Using the formula: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * (x·y)
    X_norm = np.sum(d_norm**2, axis=1)          # shape: (n_design,)
    Y_norm = np.sum(m_norm**2, axis=1)            # shape: (n_samples,)
    dot_prod = np.dot(d_norm, m_norm.T)          # shape: (n_design, n_samples)
    dist_sq = X_norm[:, None] + Y_norm[None, :] - 2 * dot_prod
    dist_sq = np.maximum(dist_sq, 0)              # Prevent negative values due to numerical errors
    dist_matrix = np.sqrt(dist_sq)                # Get the Euclidean distance matrix, shape (n_design, n_samples)

    # Implement nearest neighbor assignment without replacement using the Hungarian algorithm
    # linear_sum_assignment returns (row_ind, col_ind)
    # where the length of row_ind equals the number of design points, and each design point corresponds to a unique m point
    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    # Return the indices of the matched m (add +1 for MATLAB-style indexing if needed)
    return col_ind