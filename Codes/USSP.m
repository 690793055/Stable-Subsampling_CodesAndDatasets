%%
% Project Name: USSP
% Description: The uniform-subsampled stable prediction (USSP) algorithm
% Author: Yang Jinjing
% Email: yangjinjing94@163.com
% Date: 2024-09-14
%%

% INPUTS
%   m       : the full sample points
%   d       : the uniform design points
%
%
% OUTPUT 
%   cm      : the global stability loss of the USSP subsample points
%   id      ：the points index of the USSP subsample points

function [cm,id]=USSP(m,d)  %% m:the full sample points d:the design points
    warning off
    [drow,dcol]=size(d);
    m=normalization(m,-1,1);   %% Zoom to the range of [-1,1]
    d=normalization(d,-1,1);
    perm=perms(1:dcol);     %% permutate the column of the design
    [prow,~]=size(perm);
    if prow>1000    %% To shorten testing time, take the random 1000 permutations 
        cm_all=inf(1000,1);
        id_all=cell(1000,1);
        q=round(unifrnd(1,prow,1000,1));
        parfor_progress(1000);   %%Parallel computing
        parfor i=1:1000
            d_new=d(:,perm(q(i),:));
            Mdl = createns(m,'Distance','euclidean');   %% create the KD-tree
            id = knnsearch(Mdl,d_new);   %% KNN search by KD-tree
            m_real=m(id,:);
            cm_all(i)=GSL(m_real);    %% Compute the global stability loss
            id_all(i)={id};
        end
        num=find(cm_all==min(cm_all));
        id=id_all(num(1));
        cm=cm_all(num(1));
        parfor_progress(0);
    else
        cm_all=inf(prow,1);
        id_all=cell(prow,1);
        parfor_progress(prow);
        parfor i=1:prow
            d_new=d(:,perm(i,:));
            Mdl = createns(m,'Distance','euclidean');
            id = knnsearch(Mdl,d_new);
            m_real=m(id,:);
            cm_all(i)=GSL(m_real);
            id_all(i)={id};
        end
        num=find(cm_all==min(cm_all));
        id=id_all(num(1));
        cm=cm_all(num(1));
        parfor_progress(0);
    end
    
    
    