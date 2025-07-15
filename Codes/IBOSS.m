%% 
% Project Name: USSP
% Description: The IBOSS algorithm
% Author: Yang Jinjing
% Email: yangjinjing94@163.com
% Date: 2025-04-19
%%

% INPUTS
%   M       : the full sample points
%   k       : the number of subsample points
%
% OUTPUT 
%   id      : the indices of the IBOSS subsample points

function [id, result] = IBOSS(M, k)  %% M: the full sample points
    M_ori = M;
    [n, p] = size(M);
    id = zeros(k, 1);
    
    if mod(k, (2*p)) ~= 0
        error('k/2p is not an integer, choose another k');
    else
        result = [];
    
        for i = 1:p % Iterate over each column of the matrix
            column = M(:, i); % Get data from the current column

            % Select the indices of the top k/2p largest and smallest values
            [~, idx_max] = maxk(column, k / (2*p));
            [~, idx_min] = mink(column, k / (2*p));

            % Store the corresponding points in the result
            result = [result; M(idx_max, :); M(idx_min, :)];

            % Remove the selected points from the original matrix
            M([idx_max; idx_min], :) = []; 
        end

        for i = 1:k
            % Retrieve the original indices of the selected points
            id(i) = find(ismember(M_ori, result(i, :), 'rows'));
        end
    end
end
