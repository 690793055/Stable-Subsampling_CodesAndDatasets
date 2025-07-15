%%
% Project Name: USSP
% Description: Generate normal distribution train samples of experiment 5.3.2
% Author: Yang Jinjing
% Email: yangjinjing94@163.com
% Date: 2024-09-14
%%
% INPUTS
%   r       : the correlation of the noisy variable and the response
%   variable
%   n       : number of samples
%   seed    : seed number
%
%
% OUTPUT 
%   ss      : sample points

function [ss] = Sample_Generate_GP_NoninearlySeparableTerm(r,n,seed)
rng(seed,'twister');
beta_s=[1/3;-2/3;1;2/3;-1/3];
x=mvnrnd(zeros(1,8), eye(8), 2*n);        %% Normal distribution x
x = x(logical(prod(abs(x)<=3,2)),:);    %% Truncated x value in [-3,3]
x = x(1:n,:);
v1=x(:,6);   %%noisy variable v1
v2=x(:,7);    %%noisy variable v1
error=normrnd(0,0.3,[n,1]);
vnum=randsample(n,abs(r)*n);
diff=normrnd(0,0.1,[n,1]);
if r>0        %%create the corrleation of v1,v2 and the auxiliary variables 
    for i=1:length(vnum)
        x(vnum(i),1)=x(vnum(i),1)+1/5*v1(vnum(i))+diff(i);
        x(vnum(i),2)=x(vnum(i),2)+1/5*v1(vnum(i))+diff(i);
        x(vnum(i),3)=x(vnum(i),3)+1/5*v2(vnum(i))+diff(i);
        x(vnum(i),4)=x(vnum(i),4)+1/5*v2(vnum(i))+diff(i);
        x(vnum(i),5)=x(vnum(i),5)+1/5*v1(vnum(i))+1/5*v2(vnum(i))+diff(i);
    end
else
    for i=1:length(vnum)
        x(vnum(i),1)=x(vnum(i),1)-1/5*v1(vnum(i))+diff(i);
        x(vnum(i),2)=x(vnum(i),2)-1/5*v1(vnum(i))+diff(i);
        x(vnum(i),3)=x(vnum(i),3)-1/5*v2(vnum(i))+diff(i);
        x(vnum(i),4)=x(vnum(i),4)-1/5*v2(vnum(i))+diff(i);
        x(vnum(i),5)=x(vnum(i),5)-1/5*v1(vnum(i))-1/5*v2(vnum(i))+diff(i);
    end
end
y=x(:,1).^2+exp(x(:,2))+exp(x(:,3))+x(:,4).^2+sin(x(:,5))+x(:,1:5)*beta_s+error;  %%the response value
ss=[x,y];
