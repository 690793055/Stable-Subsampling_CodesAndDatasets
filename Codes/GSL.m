%%
% Project Name: USSP
% Description: Calculate the global stability loss of the dataset x
% Author: Yang Jinjing
% Email: yangjinjing94@163.com
% Date: 2025-04-19
%%

%%
%Input: the dataset x
%Output: the global stability loss of x

function gls = GSL(matrix)
%GSL_CALCULATE Iterates through each column of a matrix, calculates the ACC of that column with the rest,
%   and returns the mean of all ACC values.
%
%   Args:
%       matrix (double): The input matrix.
%
%   Returns:
%       gls (double): The mean of all calculated ACC values. Returns 0 if the matrix has only one column.

    num_cols = size(matrix, 2);
    if num_cols < 2
        warning('Input matrix has only one column, cannot calculate ACC of column with the rest, returning 0.');
        gls = 0.0;
        return;
    end

    acc_values = [];
    num_rows = size(matrix, 1);

    for i = 1:num_cols
        % Use the current column as U
        U = matrix(:, i);

        % Use the remaining columns as Z
        remaining_cols_indices = 1:num_cols;
        remaining_cols_indices(i) = [];
        Z = matrix(:, remaining_cols_indices);

        % Calculate the ACC of the current column and the rest
        acc = calculate_acc_matlab(U, Z);
        acc_values = [acc_values, acc];
    end

    % Calculate the mean of all ACC values
    gls = abs(mean(acc_values));
end


function acc = calculate_acc_matlab(U, Z)
%CALCULATE_ACC Calculates the Azadkia-Chatterjee Coefficient (ACC).
%
%   Args:
%       U (double): A 1D array representing samples of a random variable U.
%       Z (double): A 2D array representing samples of a random vector Z, where each row is a sample.
%
%   Returns:
%       acc (double): The calculated ACC value.

    n = length(U);
    if n ~= size(Z, 1)
        error('The number of samples in U and Z must be the same.');
    end

    R = zeros(n, 1);
    L = zeros(n, 1);
    M = zeros(n, 1, 'int32'); % Stores the index of the nearest neighbor

    % Calculate R_i and L_i
    for i = 1:n
        R(i) = sum(U <= U(i)); % Number of elements in U less than or equal to U(i)
        L(i) = sum(U >= U(i)); % Number of elements in U greater than or equal to U(i)
    end

    % Find the nearest neighbor M(i)
    for i = 1:n
        distances = pdist2(Z(i, :), Z, 'euclidean'); % Calculate the Euclidean distance between Z(i, :) and all other Z_j
        distances(i) = Inf; % Set the distance to itself to infinity to exclude itself as the nearest neighbor

        min_distance = min(distances);
        nearest_neighbors_indices = find(distances == min_distance);

        % Handle ties by randomly choosing one nearest neighbor
        randomIndex = randi(length(nearest_neighbors_indices));
        M(i) = nearest_neighbors_indices(randomIndex);
    end

    % Calculate the numerator of ACC
    numerator = 0;
    for i = 1:n
        numerator = numerator + (n * min([R(i), R(M(i))]) - L(i)^2);
    end

    % Calculate the denominator of ACC
    denominator = 0;
    for i = 1:n
        denominator = denominator + L(i) * (n - L(i));
    end

    % Avoid division by zero
    if denominator == 0
        acc = 0.0; % Or return other value or raise exception depending on the specific situation
    else
        % Calculate ACC
        acc = numerator / denominator;
    end
end








% function [y] = GSL(x) 
%     [~,m]=size(x);
%     x_squared = x.^2;
% 
%     R=cov(x);
%     y1=sum(R(:)) - trace(R);  %% First part of the the global stability loss
% 
%     y2 = 0;
%     y3 = 0;
% 
%     for i = 1:m    %% Second and third part of the the global stability loss
%         for j = 1:m
%             if i == j
%                 y2=y2+0;
%                 y3=y3+0;
%             else
%                 t2=cov(x_squared(:, i), x(:, j));
%                 t3=cov(x(:, i), x_squared(:, j));
%                 y2= y2+t2(1,2);
%                 y3= y3+t3(1,2);
%             end
%         end
%     end
% 
%     R2=cov(x_squared);
%     y4=sum(R2(:)) - trace(R2);  %% Last part of the the global stability loss
% 
%     y=(y1+y2+y3+y4)/(m^2);
