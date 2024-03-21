%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         pl
% Description:  plot figurue for TAGI
% Authors:      Luong-Ha Nguyen & James-A. Goulet 
% Created:      November 03, 2019
% Updated:      November 08, 2020
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca 
% Copyright (c) 2020 Luong-Ha nguyen  & James-A. Goulet  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef ensem
    methods (Static)        
        function [myEnsem, SyEnsem, w] = ensembleLL(y, Sy, LL)
            likelihood = exp(LL);
            w  = likelihood./sum(likelihood,2);
            myEnsem  = sum(y.*w,2);
            SyEnsem = sum(Sy.*w + w.*(y-myEnsem).^2,2);
        end
        
        function [myEnsem, SyEnsem] = ensembleWeight(y, Sy, w)
            myEnsem  = sum(y.*w,2);
            SyEnsem = sum(Sy.*w + w.*(y-myEnsem).^2,2);
        end
        
        function [myEnsem, SyEnsem] = ensembleWeight_cell(y, Sy, w)
            n = size(y,1);
            nbTs = size(y{1},2);
            nbPoint = size(y{1},1);
            ytemp  = [];
            Sytemp = [];
            for i = 1:n
               ytemp  = cat(2, ytemp, reshape(y{i}, nbPoint, 1, []));
               Sytemp = cat(2, Sytemp, reshape(Sy{i}, nbPoint, 1, []));
            end
            myEnsem  = sum(ytemp.*w,2);
            SyEnsem  = sum(Sytemp.*w + w.*(ytemp-myEnsem).^2,2);
            myEnsem  = reshape(myEnsem,nbPoint,nbTs);
            SyEnsem  = reshape(SyEnsem,nbPoint,nbTs);
        end

        function [myEnsem] = ensembleWeightBackProp_cell(y, w)
            n = size(y,1);
            nbTs = size(y{1},2);
            nbPoint = size(y{1},1);
            ytemp  = [];
            for i = 1:n
                ytemp  = cat(2, ytemp, reshape(y{i}, nbPoint, 1, []));
            end
            myEnsem  = sum(ytemp.*w,2);
            myEnsem  = reshape(myEnsem,nbPoint,nbTs);
        end
        
        function [myEnsem, SyEnsem, w] = ensemble_dirichlet(y, Sy, LL, alpha)            
            ns = 1E6; % number of Monte-Carlo samples
            nb = numel(LL);
            r  = ensem.drchrnd(alpha*ones(1,nb),ns);
            likelihood = exp(LL);
            % Posterior f(p|D)=f(D|m)*dirichlet(p;alpha)
            w  = likelihood.*r./sum(likelihood.*r,2);
            w  = mean(w);
            myEnsem  = sum(y.*w,2);
            SyEnsem = sum(Sy.*w + w.*(y-myEnsem).^2,2);
        end
       
        function r = drchrnd(a,n)
            % take a sample from a dirichlet distribution
            p = length(a);
            r = gamrnd(repmat(a,n,1),1,n,p);
            r = r ./ repmat(sum(r,2),1,p);
        end
    end
end