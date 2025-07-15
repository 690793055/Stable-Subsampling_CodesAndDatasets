%%
% Project Name: USSP
% Description: Generate normal distribution train samples of experiment 5.3.1
% Author: Yang Jinjing
% Email: yangjinjing94@163.com
% Date: 2025-04-19
%%

% INPUTS
%   r       : the correlation of the noisy variable and the response
%   variable
%   n       : number of samples
%   seed    : seed number
%   missterm : 'linear','squared' or 'exp'
%
% OUTPUT 
%   ss      : sample points

function [ss] = Sample_Generate_GP_LinearlySeparableTerm(r,n,seed,missterm)
rng(seed,'twister');
beta_s=[1;1;1;1;1];
beta_v=[0;0;0];
x=mvnrnd(zeros(1,8), eye(8), 2*n);     %% Normal distribution x
x = x(logical(prod(abs(x)<=3,2)),:);  %% Truncated x value in [-3,3]
x = x(1:n,:);
error=normrnd(0,0.3,[n,1]);
diff=normrnd(0,0.1,[n,1]);
y=x(:,1).^2+exp(x(:,2))+exp(x(:,3))+x(:,4).^2+x(:,4).*x(:,5)+x(:,1:5)*beta_s+error;   %%the response value
% m=1/4*exp(x(:,6))+1/4*exp(x(:,7));  %% The linearly-separable term
if strcmp(missterm, 'linear')
    m=x(:,6);  %% The linearly-separable term
elseif strcmp(missterm, 'squared')
    m=1/2*x(:,6).^2;  %% The squared-separable term
elseif strcmp(missterm, 'exp')
    m=1/4*exp(x(:,6))+1/4*exp(x(:,7));  %% The exp-separable term
else
    error('Invalid missterm specified. Choose "squared", "exp", or "linear".');
end
y=y+r*m+diff;
ss=[x(:,1:5),y];