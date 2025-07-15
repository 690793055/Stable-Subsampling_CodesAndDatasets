%%
% Project Name: USSP
% Description: Generate normal distribution train samples of experiment 5.2
% Author: Yang Jinjing
% Email: yangjinjing94@163.com
% Date: 2024-09-14
%%

% INPUTS
%   n       : number of samples
%   seed    : seed number
%
%
% OUTPUT 
%   ss      : sample points

function [ss] = Sample_Generate_GP_OneRegion(n,seed)
rng(seed,'twister');
cor=rand_corr_matrix(5);
X0 = mvnrnd([0,0,0,0,0], cor, 2*n);    %% Normal distribution x
X0 = X0(logical(prod(abs(X0)<=3,2)),:);   %% Truncated x value in [-3,3]
X0 = X0(1:n,:);


%% Response  value of inconsistent environment
% %%Part 1 Data Construction  -2<=x1,x2,x3,x4,x5<=2
% beta_s=[1/3;-2/3;1;2/3;-1/3];
% x1 = X0(logical(prod(abs(X0)<=2,2)),:);
% diff=normrnd(0,0.1,[length(x1),1]);
% y1=(x1(:,1).^2+x1(:,2).^2+x1(:,3).^2+x1(:,4).^2+x1(:,5).^2)...
% +5.*x1(:,1).*x1(:,2)+4.*x1(:,2).*x1(:,3)+3.*x1(:,4).*x1(:,5)+2.*x1(:,3).*x1(:,5)+x1(:,1:5)*beta_s+diff;
% 
% % %%Part 2 Data Construction    -4<=x1<=4,2<=x2<=4
% beta_s=[1/2;-3/2;1;-1/2;3/2];
% x2=X0(logical(X0(:,5)<3 & X0(:,5) >2),:);
% diff=normrnd(0,0.1,[length(x2),1]);
% y2=(x2(:,1).^3+x2(:,2).^2+x2(:,3).^3+x2(:,4).^2+x2(:,5).^3)...
%     +4.*x2(:,1).*x2(:,2)+3.*x2(:,2).*x2(:,3)+2.*x2(:,4).*x2(:,5)+5.*x2(:,3).*x2(:,5)+x2(:,1:5)*beta_s+diff;
% 
% 
% % %%Part 3 Data Construction    -4<=x1<=4,-4<=x2<=-2
% beta_s=[-1/5;-2/5;-3/5;4/5;1];
% x3=X0(logical(X0(:,5)<-2 & X0(:,5) >-3),:);
% diff=normrnd(0,0.1,[length(x3),1]);
% y3=(x3(:,1).^2+x3(:,2).^3+x3(:,3).^2+x3(:,4).^3+x3(:,5).^2)...
%     +3.*x3(:,1).*x3(:,2)+2.*x3(:,2).*x3(:,3)+5.*x3(:,4).*x3(:,5)+4.*x3(:,3).*x3(:,5)+x3(:,1:5)*beta_s+diff;
% 
% 
% % %%Part 4 Data Construction    -4<=x1<=-2,-2<=x2<=2
% beta_s=[1/5;2/5;3/5;-4/5;-1];
% x4 = X0(logical((X0(:,5)<2 & X0(:,5) >-2) & (abs(X0(:,1)) >2 | abs(X0(:,2)) >2 | abs(X0(:,3)) >2 | abs(X0(:,4)) >2)),:);
% diff=normrnd(0,0.1,[length(x4),1]);
% y4=(exp(x4(:,1))+x4(:,2).^2+exp(x4(:,3))+x4(:,4).^2+exp(x4(:,5)))...
%     +2.*x4(:,1).*x4(:,2)+5.*x4(:,2).*x4(:,3)+4.*x4(:,4).*x4(:,5)+3.*x4(:,3).*x4(:,5)+x4(:,1:5)*beta_s+diff;
% 
% 
% x=[x1;x2;x3;x4];
% y=[y1;y2;y3;y4];

%% Response  value of consistent environment
x=X0;
beta_s=[1/3;-2/3;1;2/3;-1/3];
diff=normrnd(0,0.1,[length(x),1]);
y=(x(:,1).^2+x(:,2).^2+x(:,3).^2+x(:,4).^2+x(:,5).^2)...
    +5.*x(:,1).*x(:,2)+4.*x(:,2).*x(:,3)+3.*x(:,4).*x(:,5)+2.*x(:,3).*x(:,5)+x(:,1:5)*beta_s+diff;

ss=[x,y];