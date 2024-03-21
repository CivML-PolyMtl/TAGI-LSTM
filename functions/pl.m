%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         pl
% Description:  plot figurue for TAGI
% Authors:      Luong-Ha Nguyen & James-A. Goulet 
% Created:      November 03, 2019
% Updated:      November 08, 2020
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca 
% Copyright (c) 2020 Luong-Ha nguyen  & James-A. Goulet  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef pl
    methods (Static)
        function plPrediction (t, y, ttest, ytest, Sytest, epoch, color1, color2)
            hold on
            plot(t, y, color1,'lineWidth',1)
            plot(ttest, ytest,color2,'lineWidth',1)
            pt = [ttest fliplr(ttest)];
            py = [ytest' + sqrt(Sytest') fliplr(ytest'-sqrt(Sytest'))];
            patch(pt,py,color2,'EdgeColor','none','FaceColor',color2,'FaceAlpha',0.2);
            
%             datetick('x','mm-yy');
            ylabel('y');
%             xlabel('[MM-YY]')
            iter  = char(['Epoch ' int2str(epoch)]);
            title (iter)
        end

    end
end