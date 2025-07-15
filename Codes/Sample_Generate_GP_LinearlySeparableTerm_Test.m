%%
% Project Name: USSP
% Description: Generate uniform distribution test samples of experiment 5.3.1
% Author: Yang Jinjing
% Email: yangjinjing94@163.com
% Date: 2023-07-18
%%

% INPUTS
%   r       : the correlation of the noisy variable and the response
%   variable
%   n       : number of samples
%   seed    : seed number
%   distribution :'normal' or 'uniform'
%   missterm : 'linear','squared' or 'exp'
%
%
% OUTPUT 
%   ss      : sample points

function [ss] = Sample_Generate_GP_LinearlySeparableTerm_Test(r,n,seed,distribution,missterm)
rng(seed,'twister');
beta_s=[1;1;1;1;1];
beta_v=[0;0;0];


if strcmp(distribution, 'normal')
    x=mvnrnd(ones(1,8), eye(8), n);    %% Normal distribution x
    % x = x(logical(prod(abs(x)<=3,2)),:);  %% Truncated x value in [-3,3]
    % x = x(1:n,:);
elseif strcmp(distribution, 'uniform')
    x=unifrnd(-3,3, [n,8]);    %% uniformly distribution x
else
    error('Invalid distribution specified. Choose "normal" or "uniform".');
end

error=normrnd(0,0.3,[n,1]);
diff=normrnd(0,0.1,[n,1]);
y=x(:,1).^2+exp(x(:,2))+exp(x(:,3))+x(:,4).^2+x(:,4).*x(:,5)+x(:,1:5)*beta_s+error;   %%the response value

if strcmp(missterm, 'linear')
    m=x(:,6);  %% The linearly-separable term
elseif strcmp(missterm, 'squared')
    m=1/2*x(:,6).^2;  %% The linearly-separable term
elseif strcmp(missterm, 'exp')
    m=1/4*exp(x(:,6))+1/4*exp(x(:,7));  %% The linearly-separable term
else
    error('Invalid missterm specified. Choose "squared", "exp", or "linear".');
end

y=y+r*m+diff;
ss=[x(:,1:5),y];
