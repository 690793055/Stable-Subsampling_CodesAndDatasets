%%
% Project Name: USSP
% Description: Generate random correlation coefficient matrix of multivariate normal distribution
% Author: Yang Jinjing
% Email: yangjinjing94@163.com
% Date: 2024-09-14
%%

%%
%Input  n:the number of the variables
%Output  A:correlation coefficient matrix

function A = rand_corr_matrix(n)
rng(1,'twister');
sp=0.1;  %%Matrix sparsity, closer to 1, sparser
s=0.6;   %%In terms of matrix positivity, as the value approaches 1, there are more positive elements, indicating stronger correlation in the generated matrix. When the value approaches 0.5, the probability of positive and negative elements becomes equal, indicating weaker correlation in the generated matrix.


D=rand(n,n);
D(D<sp)=0;    

B=zeros(n,n);
Q=rand(n,n);
R=rand(n,n)*s;   %%Positive and negative decision matrix
B(Q<=R)=1;
B(Q>R)=-1;

C=sqrt(D./sum(D,2)).*B;

A=C*C';

