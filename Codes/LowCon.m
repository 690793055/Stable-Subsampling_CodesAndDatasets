%%
% Project Name: LowCon
% Description: The LowCon algorithm
% Author: Yang Jinjing
% Email: yangjinjing94@163.com
% Date: 2025-04-19
%%

% INPUTS
%   m       : the full sample points
%   d       : the OLHD design points
%
%
% OUTPUT 
%   cm      : the global stability loss of the USSP subsample points
%   id      ：the points index of the USSP subsample points

function [id]=LowCon(m,d)  %% m:the full sample points d:the design points
    warning off
    m=normalization(m,-1,1);   %% Zoom to the range of [-1,1]
    d=normalization(d,-1,1);
    Mdl = createns(m,'Distance','euclidean');
    id = knnsearch(Mdl,d);


    
    
    