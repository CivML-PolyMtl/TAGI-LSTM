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
    end
end