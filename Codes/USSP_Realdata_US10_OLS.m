%%
% Project Name: USSP
% Description: main function of experiment 6.1, the OLS model in Top 10 cities weather dataset of the United States
% Author: Yang Jinjing
% Email: yangjinjing94@163.com
% Date: 2025-04-19
%%


clc
clear

% Set the folder name
folder = '..\Real dataset\Top 10 cities weather dataset of the United States\';
% get the file name list in the folder
files = dir(fullfile(folder, '*_use.csv'));
% get all the file name tables
fileNames = {files.name};

for i=1:length(fileNames)
    name=[folder,'\',fileNames{i}];
    data{i}=us10importfile(name); % Assuming us10importfile is defined elsewhere
end


%% OLS model for full data
datause=data{2}; % Assuming data{2} is the training data
train_numb=size(datause,1);
x_train=datause(:,[4,6:14]);
y_train=datause(:,5);
train_sample=[x_train,y_train]; % Combine training features and response

beta_train_all = regress(y_train,[x_train,ones(size(x_train,1),1)]);
y_train_ols=x_train*beta_train_all(1:10)+beta_train_all(11);
[GOF_train,RMSE_train]=rsquare(y_train,y_train_ols); % Assuming rsquare is defined elsewhere

%% OLS model for subsample data (Parameters)
d1_file = 'UD_500_10.csv'; % File for USSP design points
d2_file = 'OLHD_520_10.csv'; % File for LowCon design points
k_srs = 500; % Subsample size for SRS
k_iboss = 500; % Subsample size for IBOSS
subsample_size = 500; % Subsample size for USSP and LowCon

num_repetitions = 20; % Number of repetitions for averaging RMSE

% Load design points outside the test loop as they are fixed
d1 = readmatrix(fullfile(folder, d1_file), 'Range', [2 2]);
d2_lowcon = readmatrix(fullfile(folder, d2_file), 'Range', [2 2]);

% Initialize RMSE arrays to store average RMSE for each test dataset
rmse_all = zeros(9, 1);
rmse_our = zeros(9, 1);
rmse_lowcon = zeros(9, 1);
rmse_iboss = zeros(9, 1);
rmse_srs = zeros(9, 1);

%% Generate the test dataset and calculate RMSE
flag = 1;
% Indices of test datasets (assuming these correspond to data{i})
test_data_indices = [1, 7, 9, 8, 4, 10, 5, 3, 6];

for i = test_data_indices
    test_data = data{i};
    x_test = test_data(:,[4,6:14]);
    y_test = (test_data(:,5)-32)/1.8; % Normalize y_test

    % Calculate RMSE for Baseline (Full Data OLS)
    y_test_all = (x_test * beta_train_all(1:10) + beta_train_all(11) - 32) / 1.8;
    rmse_all(flag) = sqrt(mean((y_test - y_test_all).^2));

    % Initialize temporary RMSE storage for this test dataset
    temp_rmse_our = zeros(num_repetitions, 1);
    temp_rmse_lowcon = zeros(num_repetitions, 1);
    temp_rmse_iboss = zeros(num_repetitions, 1);
    temp_rmse_srs = zeros(num_repetitions, 1);

    % Repeat subsampling, training, and testing for averaging
    for rep = 1:num_repetitions
        % USSP Method
        [cm_ussp, id_our_cell] = USSP(x_train, d1);
        id_our = id_our_cell{1,1}; % Extract index from cell
        subsample_our = train_sample(id_our,:);
        x_subsample_train_our = subsample_our(:,1:10);
        y_subsample_train_our = subsample_our(:,11);
        beta_train_our = regress(y_subsample_train_our, [x_subsample_train_our, ones(size(x_subsample_train_our,1),1)]);
        y_test_our = (x_test * beta_train_our(1:10) + beta_train_our(11) - 32) / 1.8;
        temp_rmse_our(rep) = sqrt(mean((y_test - y_test_our).^2));

        % LowCon Method
        [id_lowcon] = LowCon(x_train, d2_lowcon(:,1:10));
        subsample_lowcon = train_sample(id_lowcon,:);
        x_subsample_train_lowcon = subsample_lowcon(:,1:10);
        y_subsample_train_lowcon = subsample_lowcon(:,11);
        beta_train_lowcon = regress(y_subsample_train_lowcon, [x_subsample_train_lowcon, ones(size(x_subsample_train_lowcon,1),1)]);
        y_test_lowcon = (x_test * beta_train_lowcon(1:10) + beta_train_lowcon(11) - 32) / 1.8;
        temp_rmse_lowcon(rep) = sqrt(mean((y_test - y_test_lowcon).^2));

        % IBOSS Method
        % Ensure consistent random state if needed for reproducibility of IBOSS itself across repetitions
        % If IBOSS has internal randomness, consider managing the random seed.
        [id_iboss,~] = IBOSS(x_train, k_iboss);
        subsample_iboss = train_sample(id_iboss,:);
        x_subsample_train_iboss = subsample_iboss(:,1:10);
        y_subsample_train_iboss = subsample_iboss(:,11);
        beta_train_iboss = regress(y_subsample_train_iboss, [x_subsample_train_iboss, ones(size(x_subsample_train_iboss,1),1)]);
        y_test_iboss = (x_test * beta_train_iboss(1:10) + beta_train_iboss(11) - 32) / 1.8;
        temp_rmse_iboss(rep) = sqrt(mean((y_test - y_test_iboss).^2));

        % SRS Method
        rng(rep,'twister'); % Seed for reproducibility across repetitions for SRS
        id_srs = randperm(train_numb, k_srs);
        subsample_srs = train_sample(id_srs,:);
        x_subsample_train_srs = subsample_srs(:,1:10);
        y_subsample_train_srs = subsample_srs(:,11);
        beta_train_srs = regress(y_subsample_train_srs, [x_subsample_train_srs, ones(size(x_subsample_train_srs,1),1)]);
        y_test_srs = (x_test * beta_train_srs(1:10) + beta_train_srs(11) - 32) / 1.8;
        temp_rmse_srs(rep) = sqrt(mean((y_test - y_test_srs).^2));
    end

    % Calculate the mean RMSE over repetitions for the current test dataset
    rmse_our(flag) = mean(temp_rmse_our);
    rmse_lowcon(flag) = mean(temp_rmse_lowcon);
    rmse_iboss(flag) = mean(temp_rmse_iboss);
    rmse_srs(flag) = mean(temp_rmse_srs);

    flag = flag + 1; % Increment for the next test dataset
end

%% Plotting
figure2 = figure;
X = 1:9; % X-axis values for the 9 test datasets

Y = [rmse_all, rmse_lowcon, rmse_iboss, rmse_srs, rmse_our]; % Combine RMSEs for plotting

h = bar(X, Y);

% Set colors for bars
set(h(1),'FaceColor','#FD6D5A'); % Baseline
set(h(2),'FaceColor','#6DC354'); % LowCon
set(h(3),'FaceColor','#994487'); % IBOSS
set(h(4),'FaceColor','#518CD8'); % SRS
set(h(5),'FaceColor','#FEB40B'); % USSP

xticks([1 2 3 4 5 6 7 8 9]);
set(gca,'FontSize',15)

% Assuming these are the names for the 9 test datasets
xticklabels({'Dallas','San Antonio','San Jose','San Diego','NYC','Seattle', 'Philadelphia','LA','Phoenix'});

ylabel('RMSE','FontSize',30);
xlabel('Dataset Name','FontSize',30);

legend('Baseline','LowCon','IBOSS','SRS','USSP','location','northwest','FontSize',30);

% Optional: Save the figure
% saveas(figure2, 'RMSE_Comparison_Bar_Chart.png');