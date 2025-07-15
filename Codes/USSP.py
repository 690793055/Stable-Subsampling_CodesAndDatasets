"""
%%
% Project Name: USSP
% Description: The uniform-subsampled stable prediction (USSP) algorithm
% Author: Yang Jinjing
% Email: yangjinjing94@163.com
% Date: 2025-04-19
%%

% INPUTS
%   m       : the full sample points
%   d       : the uniform design points
%
%
% OUTPUT 
%   cm      : the global stability loss of the USSP subsample points
%   id      ：the points index of the USSP subsample points

"""
import numpy as np
from sklearn.neighbors import KDTree
from itertools import permutations
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
import GSL
import random
from scipy.optimize import linear_sum_assignment

def normalization(data, min_val, max_val):
    """Data normalization (consistent with MATLAB range)"""
    scaler = MinMaxScaler(feature_range=(min_val, max_val))
    return scaler.fit_transform(data)



def process_permutation(p,m,d):
    """Encapsulates the processing logic for a single permutation"""
    d_new = d[:, p]
    tree = KDTree(m, metric='euclidean')
    dist, id = tree.query(d_new, k=1)
    m_real = m[id.flatten(), :]
    return GSL.GSL_calculate(m_real), id


def process_permutation_without_rep(p,m,d):
    """Encapsulates the processing logic for a single permutation: nearest neighbor assignment without replacement
    Args:
        p: A permutation (list of indices) used to select columns of d
    Returns:
        Calculation result and indices of matched m (each d_new has a unique correspondence)
    """
    # Select a subset of columns from d according to the permutation p, resulting in a new feature matrix d_new
    # Convert to float32 (if data allows) to reduce memory usage
    d_new = d[:, p].astype(np.float32)
    # Convert m to float32 as well (if not already converted)
    m_float = m.astype(np.float32)

    # Calculate the distance matrix using the Euclidean distance formula (without using broadcasting to create large arrays)
    # Calculate the sum of squares for each d_new vector, shape (n,)
    X_norm = np.sum(d_new**2, axis=1)  # shape: (n,)
    # Calculate the sum of squares for each m vector, shape (N,)
    Y_norm = np.sum(m_float**2, axis=1)  # shape: (N,)
    # Calculate the dot product matrix, shape (n, N)
    dot_prod = np.dot(d_new, m_float.T)

    # Calculate the squared distance matrix, shape (n, N)
    # This avoids creating an intermediate array of shape (n, N, feat_dim)
    dist_sq = X_norm[:, None] + Y_norm[None, :] - 2 * dot_prod
    # Numerical errors might lead to negative values, take the maximum to ensure non-negativity
    dist_sq = np.maximum(dist_sq, 0)
    # Calculate the distance matrix
    dist_matrix = np.sqrt(dist_sq)

    # Use the Hungarian algorithm to obtain the optimal matching, ensuring each d_new is assigned to a unique m
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    m_real = m_float[col_ind]

    # Call GSL_calculate for subsequent calculations on the matched m_real
    return GSL.GSL_calculate(m_real), col_ind


def USSP(m, d):
    np.random.seed(1)

    # Normalization
    m = normalization(m, -1, 1)
    d = normalization(d, -1, 1)

    # Determine matrix dimension
    dim = d.shape[1]
    if dim >= 2:
        # Randomly generate 1000 permutations
        random.seed(11)
        selected_perms = [random.sample(range(dim), dim) for _ in range(10)]
    else:
        # Generate all permutations
        selected_perms = list(permutations(range(dim)))

    # Use Joblib for parallel computation
    results = Parallel(n_jobs=-1, verbose=1)(
        delayed(process_permutation)(p,m,d)
        for p in selected_perms
    )
    # Unpack results
    cm_all = np.array([res[0] for res in results])
    id_all = [res[1] for res in results]
    # Find the index of the minimum value
    min_idx = np.argmin(cm_all)
    return cm_all[min_idx], id_all[min_idx]