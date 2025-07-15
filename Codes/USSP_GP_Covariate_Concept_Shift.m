%%
% Project Name: USSP
% Description: main function of experiment 5.3
% Author: Yang Jinjing
% Email: yangjinjing94@163.com
% Date: 2025-07-10
%%

clear;
clc;

nDims = 10;          % Dimension number
nTotalPoints = 100000; % Total number of points in the data stream
nTrainSamples = 2000;  % Number of subsamples for training
nTestPoints = 2000;    % Number of test points
nRepeats = 10;         % Number of repetitions for each experiment
% Input range for Ackley function training data
x_train_min = -20;
x_train_max = 20;
% Parameters for the multi-dimensional normal distribution of the original data stream (mean and covariance matrix)
mu_original = zeros(1, nDims);
Sigma_original = (7)^2 * eye(nDims); % Assuming standard deviation is 7, to cover the [-20, 20] range
% DACE Model Parameters
reg = 'regpoly0';    % Regression model (basis function): Constant regression
corr = 'corrgauss';  % Correlation function (kernel function): Gaussian (squared exponential) kernel function
% Initial hyperparameters theta and their bounds (for 'gauss' kernel function, p parameter is fixed to 2)
theta0 = 1 * ones(1, nDims); % Initial theta values
lob = 1e-2 * ones(1, nDims);  % Lower bound for theta
upb = 10 * ones(1, nDims);    % Upper bound for theta

% Covariate shift parameter: Mean shift amount (mu_shift)
mu_shift_values = 0:1:20; % Mean shift from 0 to 20, step size is 1

% --- Modification: Covariance matrix scaling factor, same quantity as mean shift values ---
% Assuming covariance scaling factors also match the number of mean shift steps
% For example, if mu_shift_values has 21 values, sigma_scale_values should also have 21 values
% Here, I set sigma_scale_values to be a linearly increasing sequence,
% with the same length as mu_shift_values. Start and end values can be adjusted.
sigma_scale_values = linspace(1.0, 2.0, length(mu_shift_values));
% random_sigma_perturbation_strength = 0.05; % Controls the degree of random perturbation of the covariance matrix in each repeated experiment (adjustable)
random_sigma_perturbation_strength = 0.1; % Controls the degree of random perturbation of the covariance matrix in each repeated experiment (adjustable)
% --- End Modification ---

% Concept shift parameter: Fixed p values
concept_p_values = [0.1, 0.2, 0.3, 0.4]; % Four fixed concept shift intensity p values
z_std_dev = 0.25; % Standard deviation of Z(x)

%% 3. Generate Original Data (Basis for Training Data)
x_train = mvnrnd(mu_original, Sigma_original, nTotalPoints);
x_train = max(min(x_train, x_train_max), x_train_min); % Clip to training data range
fprintf('Original data generation complete.\n');

%% 4. Calculate Original Responses (Using Ackley Function)
fprintf('Calculating response values for the original data stream (Ackley function)...\n');
y_train = zeros(nTotalPoints, 1);
for k = 1:nTotalPoints
    y_train(k) = ackleyfcn(x_train(k,:));
end
fprintf('Response value calculation complete.\n');

%% 5. Subsampling
load("UD_2000_10.mat");    %%Load the uniform design points
load("LHD_2000_10.mat");
[~,id_our]=USSP_new(x_train,d1);
[id_lowcon]=LowCon_new(x_train,d2); % Add a k_neighbors_for_kd_tree parameter, e.g., 5
% [id_iboss,~]=IBOSS(x_train,nTrainSamples);

X_CAO_all = x_train;
X_CAO_all_norm_squared = sum(X_CAO_all.^2, 2);
[~,sorted_X_CAO_all_norm_squared_indices] = sort(X_CAO_all_norm_squared, 'descend');
id_iboss = sorted_X_CAO_all_norm_squared_indices(1:nTrainSamples);

rng(42,'twister'); % Ensure reproducibility of SRS sampling
id_srs = randperm(nTotalPoints, nTrainSamples);

x_train_our=x_train(id_our,:);
x_train_lowcon=x_train(id_lowcon,:);
x_train_iboss=x_train(id_iboss,:);
x_train_srs=x_train(id_srs,:);

y_train_our=y_train(id_our,:);
y_train_lowcon=y_train(id_lowcon,:);
y_train_iboss=y_train(id_iboss,:);
y_train_srs=y_train(id_srs,:);

%% 6. Train Gaussian Process Models (Using DACE)
fprintf('\nTraining Gaussian Process models (using DACE)...\n');
dmodels = struct();
[dmodels.our, ~] = dacefit(x_train_our, y_train_our, reg, corr, theta0, lob, upb);
[dmodels.lowcon, ~] = dacefit(x_train_lowcon, y_train_lowcon, reg, corr, theta0, lob, upb);
[dmodels.iboss, ~] = dacefit(x_train_iboss, y_train_iboss, reg, corr, theta0, lob, upb);
[dmodels.srs, ~] = dacefit(x_train_srs, y_train_srs, reg, corr, theta0, lob, upb);

%% 7. Simulate Covariate Shift and Concept Shift and Test
fprintf('\nSimulating compound shift (covariate mean shift + covariance shift + concept shift) and performing prediction and evaluation...\n');
% Store MSPE results for all models (for each concept shift intensity p)
mspe_results_per_concept_p = struct();
model_names = fieldnames(dmodels);

% Ensure the number of mean shifts and covariance shifts are the same
if length(mu_shift_values) ~= length(sigma_scale_values)
    error('The number of mean shift values (mu_shift_values) and covariance scaling values (sigma_scale_values) must be the same for one-to-one correspondence.');
end
nCombinedShiftSteps = length(mu_shift_values); % Number of combined shift steps

% Outer loop: Concept shift (p)
for p_idx = 1:length(concept_p_values)
% for p_idx = 1:1
    current_concept_p = concept_p_values(p_idx);
    fprintf('\n=========================================================\n');
    fprintf('Starting simulation for concept shift p = %.2f\n', current_concept_p);
    fprintf('=========================================================\n');

    % Initialize MSPE storage for the current concept shift p value
    % MSPE storage dimension is now (number of combined shift steps)
    mspe_results_per_concept_p(p_idx).p_value = current_concept_p;
    for m_name_idx = 1:length(model_names)
        current_model_name = model_names{m_name_idx};
        % Create an (nCombinedShiftSteps x 1) NaN vector to store MSPE
        mspe_results_per_concept_p(p_idx).mspe.(current_model_name) = NaN(nCombinedShiftSteps, 1);
    end

    % Single loop: Simultaneously handle mean shift and covariance shift
    for shift_idx = 1:nCombinedShiftSteps
        current_mu_shift = mu_shift_values(shift_idx);
        current_sigma_scale = sigma_scale_values(shift_idx);

        fprintf('   Processing combined covariate shift: Mean=%.2f, Covariance Scale=%.2f (Step %d/%d)...\n', ...
            current_mu_shift, current_sigma_scale, shift_idx, nCombinedShiftSteps);

        % Store MSPE results for all repetitions under the current (p, combined shift) combination
        current_combo_mspe_temp = struct();
        for m_name_idx = 1:length(model_names)
            current_combo_mspe_temp.(model_names{m_name_idx}) = NaN(nRepeats, 1);
        end

        % Under the current (p, combined shift) combination, repeat nRepeats tests
        for repeat_idx = 1:nRepeats
            % --- Generate test set input data (covariate shift - mean + covariance) ---
            % 1. Calculate shifted mean
            mu_test = mu_original + current_mu_shift * ones(1, nDims);

            % 2. Calculate shifted covariance matrix
            % First apply the scaling factor
            % Sigma_scaled = Sigma_original * current_sigma_scale;
            Sigma_scaled = Sigma_original ;
            % Then add random perturbation
            % Generate new random perturbation for each repetition, ensuring the matrix is symmetric and positive definite
            % Note: The rng seed here needs to consider the joint index of mu_shift and sigma_scale
            rng(p_idx * 100000 + shift_idx * 100 + repeat_idx, 'twister'); % Ensure reproducibility of randomness

            % Generate a random symmetric matrix as perturbation
            rand_mat = randn(nDims, nDims);
            rand_perturbation = (rand_mat + rand_mat') / 2; % Make it symmetric
            rand_perturbation = rand_perturbation * random_sigma_perturbation_strength; % Control perturbation strength

            % Add the perturbation to the scaled covariance matrix
            Sigma_test_shifted = Sigma_scaled + rand_perturbation;
            % Sigma_test_shifted = Sigma_scaled;
            % Ensure Sigma_test_shifted is positive definite
            [~, p_check] = chol(Sigma_test_shifted);
            if p_check ~= 0
                % Matrix is not positive definite, possibly due to large perturbation.
                % Increase diagonal elements to make it positive definite
                Sigma_test_shifted = Sigma_test_shifted + (abs(min(eig(Sigma_test_shifted))) + 1e-6) * eye(nDims);
            end

            X_test_shifted = mvnrnd(mu_test, Sigma_test_shifted, nTestPoints);

            % Calculate original Ackley response values for the test set (based on shifted X_test_shifted)
            y_ackley_original = zeros(nTestPoints, 1);
            for k = 1:nTestPoints
                y_ackley_original(k) = ackleyfcn(X_test_shifted(k,:));
            end

            % Generate Z(x) values (normally distributed with mean 0 and standard deviation z_std_dev)
            % Z(x) will be different but reproducible for each repetition and each (p, combined shift) combination
            Z_values = normrnd(0, z_std_dev, nTestPoints, 1);

            % Calculate true response values with multiplicative concept shift
            % y_true = y_original * (1 + p * Z(x))
            y_test_true_shifted = y_ackley_original .* (1 + current_concept_p * Z_values);

            for m_idx = 1:length(model_names)
                current_model_name = model_names{m_idx};
                current_dmodel = dmodels.(current_model_name);
                if ~isempty(current_dmodel)
                    % Predict based on shifted X_test_shifted
                    [y_pred, ~] = predictor(X_test_shifted, current_dmodel);
                    current_combo_mspe_temp.(current_model_name)(repeat_idx) = mean((y_test_true_shifted - y_pred).^2);
                end
            end
        end % End nRepeats loop

        % Calculate and store the average MSPE for all models under the current (p, combined shift) combination
        for m_idx = 1:length(model_names)
            current_model_name = model_names{m_idx};
            valid_mspe_repeats = current_combo_mspe_temp.(current_model_name)(~isnan(current_combo_mspe_temp.(current_model_name)));
            if ~isempty(valid_mspe_repeats)
                mspe_results_per_concept_p(p_idx).mspe.(current_model_name)(shift_idx) = mean(valid_mspe_repeats);
            end
        end
    end % End combined shift loop (shift_idx)
end % End concept_p_values loop
fprintf('\nAll compound shift simulations and repetitions complete.\n');

%% 8. Result Comparison and Plotting
fprintf('\n--- Result Comparison and Plotting ---\n');
% Plot line graphs for each concept shift p value
for p_idx = 1:length(concept_p_values)
% for p_idx = 1:1
    current_concept_p = mspe_results_per_concept_p(p_idx).p_value;
    current_mspe_data = mspe_results_per_concept_p(p_idx).mspe;
    figure_handle = figure;

    x_plot = 1:(nCombinedShiftSteps); % x-axis steps for plotting

    % Create a combined label containing mean and covariance scaling information
    x_axis_labels_combined = cell(1, nCombinedShiftSteps);
    for k = 1:nCombinedShiftSteps
        x_axis_labels_combined{k} = mu_shift_values(k);
    end

    plot(x_plot, sqrt(current_mspe_data.lowcon), '*-','Color','#6DC354','linewidth',2,'MarkerSize', 10);
    hold on;
    plot(x_plot, sqrt(current_mspe_data.iboss), 'pentagram-','Color','#994487','linewidth',2,'MarkerSize', 10);
    plot(x_plot, sqrt(current_mspe_data.srs), 'd-','Color','#518CD8','linewidth',2,'MarkerSize', 10);
    plot(x_plot, sqrt(current_mspe_data.our), '^-','Color','#FEB40B','linewidth',2,'MarkerSize', 10);

    current_axes = gca;
    current_axes.XLim = [min(x_plot), max(x_plot)];
    set(current_axes, 'xTick', x_plot, 'xTickLabel', x_axis_labels_combined, ...
                        'FontSize', 18); % Rotate labels to prevent overlap
    hold off;
    xlabel('Covariate Shift Magnitude','FontSize',25);
    ylabel('RMSE','FontSize',25);
    title_str = sprintf('Concept Shift t = %.2f', current_concept_p);
    title(title_str, 'FontSize', 25);
    lgd = legend('LowCon','IBOSS','SRS','USSP','FontSize',20,'Location', 'north');
end
fprintf('\nAll result plots generated.\n');

%% 9. Sample Distribution Comparison Plot (2D Projection - Combined Subplots)
fprintf('\n--- Plotting Sample Distribution Comparison (2D Projection - Combined Subplots) ---\n');

% Select dimensions for 2D projection
dim1_idx = 1; % First dimension
dim2_idx = 2; % Second dimension

% Check if dimensions are valid
if size(x_train, 2) < max(dim1_idx, dim2_idx)
    warning('Insufficient data dimensions for 2D projection. x_train has %d columns, but attempting to use dimensions %d and %d. Please adjust dim1_idx and dim2_idx.', size(x_train, 2), dim1_idx, dim2_idx);
else
    % Extract 2D projected data of the original full sample
    x_train_2d_full = x_train(:, [dim1_idx, dim2_idx]);

    % Define plotting configurations
    sample_plot_configs = struct(...
        'our',   struct('name', 'USSP',   'X', x_train_our,   'color', '#FEB40B', 'marker', '^'), ... % USSP (yellow/orange tone)
        'iboss', struct('name', 'IBOSS',  'X', x_train_iboss, 'color', '#994487', 'marker', 'p'), ... % IBOSS (purple tone)
        'lowcon',struct('name', 'LowCon', 'X', x_train_lowcon,'color', '#6DC354', 'marker', '*'), ... % LowCon (green tone)
        'srs',   struct('name', 'SRS',    'X', x_train_srs,   'color', '#518CD8', 'marker', 'd')     ... % SRS (blue tone)
    );

    % Adjust point sizes, may need to be larger in subplots for visibility
    marker_size_full = 3;   % Marker size for original full sample (reduced)
    marker_size_sampled = 6; % Marker size for sampled points (adjusted appropriately)

    % Create a new figure to place all subplots
    figure('Name', 'Sample Distribution Comparison (2D Projection)', 'Position', [100, 100, 1000, 800]); % Adjust figure size to accommodate subplots

    % Get key names of all sampling methods
    method_keys = fieldnames(sample_plot_configs);

    % Loop to plot each subplot
    for i = 1:length(method_keys)
        current_key = method_keys{i};
        method_info = sample_plot_configs.(current_key);
        method_name = method_info.name;
        x_sampled_current = method_info.X;
        sampled_color_hex = method_info.color;
        sampled_marker = method_info.marker;


        % Create subplot: 2 rows, 2 columns, current is the i-th subplot
        subplot(2, 2, i);
        hold on;

        % Plot original full sample as background
        plot(x_train_2d_full(:, 1), x_train_2d_full(:, 2), 'o', ...
             'MarkerSize', marker_size_full, ...
             'MarkerEdgeColor', [0.7 0.7 0.7], ...
             'MarkerFaceColor', [0.8 0.8 0.8]);

        % Plot samples for the current sampling method
        if ~isempty(x_sampled_current)
            if size(x_sampled_current, 2) >= max(dim1_idx, dim2_idx)
                plot(x_sampled_current(:, dim1_idx), x_sampled_current(:, dim2_idx), sampled_marker, ...
                     'MarkerSize', marker_size_sampled, ...
                     'MarkerEdgeColor', sampled_color_hex, ...
                     'MarkerFaceColor', sampled_color_hex, ...
                     'LineWidth', 1.0); % Maintain moderate line width
            else
                warning('Sample data dimensions for method %s are insufficient for 2D projection.', method_name);
            end
        else
            warning('No sample data extracted for method %s, skipping plot.', method_name);
        end

        hold off;

        xlabel(sprintf('Dim %d', dim1_idx), 'FontSize', 10); % Reduce font size to fit subplot
        ylabel(sprintf('Dim %d', dim2_idx), 'FontSize', 10);
        title(method_name, 'FontSize', 12); % Subplot title only shows method name

        % Add subplot legend
        legend({'Full Data', [method_name ' Samples']}, 'Location', 'best', 'FontSize', 8); % Reduce font size
        grid on;
        axis tight;
    end

    % Add a main title for the entire figure (optional)
    sgtitle(sprintf('Sample Distribution Comparison (Dims %d & %d)', dim1_idx, dim2_idx), 'FontSize', 16, 'FontWeight', 'bold');

end

fprintf('Sample distribution comparison plots generated.\n');