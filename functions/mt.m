%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         mt
% Description:  Metric for performance evaluation
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      November 8, 2019
% Updated:      April 04, 2020
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2020 Luong-Ha nguyen & James-A. Goulet
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef mt
    methods (Static)
        function e = computeError(y, ypred)
            e = mean(sqrt(mean((y-ypred).^2)));
        end
        function e = computeRMSE(y, ypred)
            ypred = double(ypred);
            e = reshape((y-ypred).^2,[],1);
            e = sqrt(mean(e));
        end
        function MASE = computeMASE(y, ypred, ytrain, seasonality)
            nbts = size(y,2);
            se = nan(1,size(y,2));
            for i = 1:nbts
                ytrain_ = ytrain(:,i);
                ytrain_(isnan(ytrain_)) = [];
                se(i) = mean(abs(ytrain_(seasonality+1:end) - ytrain_(1:end-seasonality)));
            end
            se(find(se ==0)) = nan; 
            MASE = mean(abs(y-ypred))./se;
            MASE = nanmean(MASE);
        end
        function ND = computeND(y, ypred)
            ND = sum(abs(ypred-y),'all')/sum(abs(y),'all');
        end
        function QL = compute90QL (y, ypred, Vpred)
            ypred_90q = ypred + 1.282*sqrt(Vpred);
            Iq  =  y > ypred_90q;
            Iq_ =  y <= ypred_90q;
            e = y-ypred_90q;
            QL = sum(2.*e.*(0.9.*Iq - (1-0.9).*Iq_),'all')/sum(abs(y),'all');
        end
        function MSE = computeMSE(y, ypred)
            MSE = mean((y-ypred).^2,'all');
        end
        function [MAPE1, MAPE2]  = computeMAPE(y, ypred)
            H = size(y,1);
            MAPE1 = mean(sum(abs((y-ypred)./y))*100./H);
            MAPE2 = mean(abs((y-ypred)./y),'all')*100;
        end
        function e  = errorRate(ytrue, ypred)
            idx_true     = ytrue;
            [~,idx_pred] = max(ypred);
            idx_pred     = idx_pred-1;
            e            = idx_true~=idx_pred;  
        end
        function LL = loglik(y, ypred, Vpred)
            idx = ~isnan(y);
            y = y(idx);
            ypred = ypred(idx);
            Vpred = Vpred(idx);
            d = size(y, 2);
            if d == 1
                LL = sum(-0.5*log(2*pi*Vpred) - (0.5*(y-ypred).^2)./Vpred);
            else
                LL = sum(-d/2*log(2*pi) - 0.5*log(prod(Vpred, 2)) - sum((0.5*(y-ypred).^2)./Vpred, 2));
            end
        end
        function weight     = probFromloglik(loglik)
            maxlogpdf       = max(loglik);
            w_1             = bsxfun(@minus,loglik,maxlogpdf);
            w_2             = log(sum(exp(w_1)));
            w_3             = bsxfun(@minus,w_1,w_2);
            weight          = exp(w_3);
        end
        function NLL = negLoglik(p, labels)
            Nobs    = length(labels);
            classes = colon(min(labels), max(labels));
            y1hot   = repmat(classes, [Nobs, 1]);
            y1hot   = y1hot==labels;
            NLL     = -sum(log(p(y1hot)))/Nobs;
        end
    end
end