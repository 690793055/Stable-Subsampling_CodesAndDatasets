""""
%%
% Project Name: USSP
% Description: The uniform-subsampled stable prediction (USSP) algorithm
% Author: Yang Jinjing
% Email: yangjinjing94@163.com
% Date: 2025-04-19
%%

% INPUTS
%   M       : the full sample points
%   k       : the subsample points number
%
% OUTPUT 
%   id      ：the points index of the IBOSS subsample points
"""
import numpy as np

def IBOSS(M, k):
    M_ori = M.copy()  # Keep the original matrix
    n, p = M.shape
    if k % (2 * p) != 0:
        raise ValueError(f"k/(2p) must be an integer, current k={k}, p={p}")

    selected_original_indices = set()
    remaining_indices = np.arange(n)  # Track the original indices of the remaining samples
    M_current = M.copy()

    for i in range(p):
        if M_current.size == 0:  # Terminate early if the matrix is empty
            break

        if M_current.shape[1] <= i: # Handle cases where p might change due to empty M
            break

        column = M_current[:, i]
        m = k // (2 * p)  # Select m maximum/minimum values per column

        # Get the indices of the maximum values in the current M
        max_indices_current = np.argpartition(-column, m)[:m]
        # Get the indices of the minimum values in the current M
        min_indices_current = np.argpartition(column, m)[:m]

        # Get the original indices of these selected rows
        original_max_indices = remaining_indices[max_indices_current]
        original_min_indices = remaining_indices[min_indices_current]

        # Add these original indices to the set
        selected_original_indices.update(original_max_indices)
        selected_original_indices.update(original_min_indices)

        # Update remaining samples (remove selected rows)
        indices_to_remove_current = np.union1d(max_indices_current, min_indices_current)
        mask = np.ones(len(M_current), dtype=bool)
        mask[indices_to_remove_current] = False
        M_current = M_current[mask]
        remaining_indices = remaining_indices[mask]

    # Convert the set of original indices to a list and take the first k
    selected_indices_list = sorted(list(selected_original_indices))[:k]
    id = np.array(selected_indices_list)
    result = M_ori[id]

    return id, result