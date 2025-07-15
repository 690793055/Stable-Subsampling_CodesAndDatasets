%%
% Project Name: USSP
% Description: Generate normal distribution train samples of experiment 5.1
% Author: Yang Jinjing
% Email: yangjinjing94@163.com
% Date: 2025-04-19

% INPUTS
%   r       : the correlation of the noisy variable and the response
%   variable
%   n       : number of samples
%   seed    : seed number
%   missterm : 'squared', 'exp', or 'empty'
%
% OUTPUT 
%   ss      : sample points
%   beta_s  ：beta value of stale variables s
%   beta_v  ：beta value of noisy variables v
 


function [ss,beta_s,beta_v] = Sample_Generate_OLS_8dimension(r,n,seed,missterm)

rng(seed+2,'twister');
% beta_s=[1/3;-2/3;1;2/3;-1/3];
beta_s=[1;1;1;1;1];
beta_v=[0;0;0];
% sigma=rand_corr_matrix(6);
% x=mvnrnd(zeros(1,6), sigma,2*n);
x=mvnrnd(zeros(1,6), eye(6),2*n);
x = x(logical(prod(abs(x)<=3,2)),:);
x = x(1:n,:);
% x = unifrnd(-3,3, [n,6]);
%x=unifrnd(-3,3,n,3);
error=normrnd(0,0.3,[n,1]);
% y=x(:,1:5)*beta_s+error;
if strcmp(missterm, 'squared')
    y=1/5*(x(:,1).^2+x(:,4).^2+x(:,5).^2)+x(:,1:5)*beta_s+error;
elseif strcmp(missterm, 'exp')
    y=1/5*(exp(x(:,1))+exp(x(:,2))+exp(x(:,5)))+x(:,1:5)*beta_s+error;
elseif strcmp(missterm, 'empty')
    y=x(:,1:5)*beta_s+error;
else
    error('Invalid missterm specified. Choose "squared", "exp", or "empty".');
end
% y=1/5*(exp(x(:,1))+exp(x(:,2))+exp(x(:,5)))+x(:,1:5)*beta_s+error;
% y=1/5*x(:,1).*x(:,2).*x(:,3).*x(:,5)+x(:,1:5)*beta_s+error;
% v=mvnrnd(zeros(1,1), eye(1), n);
% v=mvnrnd(zeros(1,2), eye(2), n);
v=mvnrnd(zeros(1,2), eye(2), n);
vnum=randsample(n,abs(r)*n);
% vall=[1:n]';
% velse=setdiff(vall,vnum);
diff=normrnd(0,0.1,[length(vnum),1]);
% diff=zeros(length(vnum),1);
if r>0
    for i=1:length(vnum)
        v(vnum(i),1)=1/2*y(vnum(i))+diff(i);
        v(vnum(i),2)=1/3*y(vnum(i))+diff(i);
    end
else
    for i=1:length(vnum)
        v(vnum(i),1)=-1/2*y(vnum(i))+diff(i);
        v(vnum(i),2)=-1/3*y(vnum(i))+diff(i);
    end
end
ss=[x,v,y];