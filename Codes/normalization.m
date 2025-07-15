%%
% Project Name: USSP
% Description: Method for normalizing data x by column to any interval [ymin, ymax] range
% Author: Yang Jinjing
% Email: yangjinjing94@163.com
% Date: 2024-09-14
%%
%%
%Input
%x: data that needs to be normalized
%ymin: lower limit of normalized interval [ymin, ymax]
%ymax: upper limit of normalized interval [ymin, ymax]

%Output
%y: the normalized data
function [ y ] = normalization( x,ymin,ymax )

[m,n]=size(x);
y=zeros(m,n);
for i=1:n
    xx=x(:,i);
    xmax=max(xx);

    xmin=min(xx);

    y(:,i)= (ymax-ymin)*(xx-xmin)/(xmax-xmin) + ymin;
end



end