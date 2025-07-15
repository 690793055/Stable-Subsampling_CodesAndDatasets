%%
% Project Name: USSP
% Description: Toy example figure of Figure 1 
% Author: Yang Jinjing
% Email: yangjinjing94@163.com
% Date: 2025-04-19
%%


% Set parameters
n = 1000; % Number of discretization points
x = linspace(0, 1, n); % Define the range of the x-axis
% Function pi: Uniform distribution on 0-1
pdf_pi = zeros(size(x));
pdf_pi(x >= 0 & x <= 1) = 1; % Probability density of uniform distribution on [0, 1] is 1
% Function A: Truncated normal distribution, mean at the center, small standard deviation
mean_A = 0.6;
std_A = 0.2;
pdf_A_untruncated = normpdf(x, mean_A, std_A);
pdf_A = pdf_A_untruncated;
pdf_A(x < 0 | x > 1) = 0; % Set values outside the range [0, 1] to 0
normalization_A = trapz(x, pdf_A); % Calculate the integral after truncation
pdf_A = pdf_A / normalization_A; % Normalize so that the integral is approximately 1
% Function B: Truncated normal distribution, mean near the boundary, very small standard deviation
mean_C = 0.2;
std_C = 0.1;
pdf_C_untruncated = normpdf(x, mean_C, std_C);
pdf_C = pdf_C_untruncated;
pdf_C(x < 0 | x > 1) = 0; % Set values outside the range [0, 1] to 0
normalization_C = trapz(x, pdf_C); % Calculate the integral after truncation
pdf_C = pdf_C / normalization_C; % Normalize so that the integral is approximately 1
% Function C: Multimodal distribution (mixture of two Gaussian distributions)
mean_B1 = 0.8;
std_B1 = 0.08;
weight_B1 = 0.6; % Weight of the first Gaussian distribution
mean_B2 = 0.45;
std_B2 = 0.12;
weight_B2 = 0.4; % Weight of the second Gaussian distribution
pdf_B_untruncated = weight_B1 * normpdf(x, mean_B1, std_B1) + weight_B2 * normpdf(x, mean_B2, std_B2);
pdf_B = pdf_B_untruncated;
pdf_B(x < 0 | x > 1) = 0; % Set values outside the range [0, 1] to 0
normalization_B = trapz(x, pdf_B); % Calculate the integral after truncation
pdf_B = pdf_B / normalization_B; % Normalize so that the integral is approximately 1
% Morandi color scheme
color_pi = [0.7, 0.7, 0.7];       % Light gray
color_A = [0.6, 0.8, 0.6];       % Light green
color_C = [0.9, 0.6, 0.6];       % Light red
color_B = [0.5, 0.7, 0.9];       % Light blue
% Plot the function images
figure;
h1=plot(x, pdf_pi, 'Color', color_pi, 'LineWidth', 3, 'DisplayName', 'Uniform measure $\pi$');
hold on;
h2=plot(x, pdf_A, 'Color', color_A, 'LineStyle', '--', 'LineWidth', 3, 'DisplayName', 'Probability measure $\eta_A$'); % Modified here
h3=plot(x, pdf_B, 'Color', color_B, 'LineStyle', ':', 'LineWidth', 3, 'DisplayName', 'Probability measure $\eta_B$'); % Plot C
h4=plot(x, pdf_C, 'Color', color_C, 'LineStyle', '-.', 'LineWidth', 3, 'DisplayName', 'Probability measure $\eta_C$'); % Modified here
uistack(h1, 'top');
hold off;
set(gca,'FontSize',15)
xlabel('x','FontSize',25);
ylabel('Probability density','FontSize',25);
legend([h1, h2, h3, h4],'FontSize',20, 'Interpreter', 'latex','NumColumns',1); % Updated legend
grid on;
% Calculate the total variation between A and pi (calculated over the entire domain)
tv_piA = 0.5 * trapz(x, abs(pdf_A - pdf_pi));
fprintf('Total Variation between function A and function pi (entire domain): %.4f\n', tv_piA);
% Calculate the total variation between B and function pi (calculated over the entire domain)
tv_piB = 0.5 * trapz(x, abs(pdf_C - pdf_pi));
fprintf('Total Variation between function C and function pi (entire domain): %.4f\n', tv_piB);
% Calculate the total variation between C and function pi (calculated over the entire domain)
tv_piC = 0.5 * trapz(x, abs(pdf_B - pdf_pi));
fprintf('Total Variation between function B and function pi (entire domain): %.4f\n', tv_piC);
% Display the total variation values on the plot
max_pdf_A = max(pdf_A);
text_y_pos_A = max_pdf_A * 1.1; % Slightly above the peak of A
tv_piA_str = sprintf('TV($\\pi$, $\\eta_A$) = %.4f', tv_piA);
text(mean_A, text_y_pos_A, tv_piA_str, 'FontSize', 20, 'Interpreter', 'latex', 'HorizontalAlignment', 'center');
max_pdf_C = max(pdf_C);
text_y_pos_C = max_pdf_C * 1.05; % Slightly above the peak of B
tv_piB_str = sprintf('TV($\\pi$, $\\eta_C$) = %.4f', tv_piB);
text(mean_C, text_y_pos_C, tv_piB_str, 'FontSize', 20, 'Interpreter', 'latex', 'HorizontalAlignment', 'center');
% Display the total variation value of C on the plot
% Choose a suitable position to display the total variation value of C
mean_B_Bisplay = (weight_B1+0.1) * mean_B1 + (weight_B2-0.1) * mean_B2; % Weighted average as a reference for display position
max_pdf_B = max(pdf_B);
text_y_pos_B = max_pdf_B * 1.05; % Slightly above the peak of C
tv_piC_str = sprintf('TV($\\pi$, $\\eta_B$) = %.4f', tv_piC);
text(mean_B_Bisplay, text_y_pos_B, tv_piC_str, 'FontSize', 20, 'Interpreter', 'latex', 'HorizontalAlignment', 'center');