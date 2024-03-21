%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         dp
% Description:  data processing
% Authors:      James-A. Goulet & Luong-Ha Nguyen
% Created:      November 8, 2019
% Updated:      September 15, 2020
% Contact:      james.goulet@polymtl.ca & luongha.nguyen@gmail.com 
% Copyright (c) 2020 James-A. Goulet & Luong-Ha Nguyen
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef dp
    methods (Static)
        % Data
        function x = dataLoader(x, da, B, rB, trainMode)
            if da.enable == 1 && trainMode == 1
                for n = 1:B:rB*B
                    idx = n:n+B-1;
                    for k = 1:length(da.types)
                        if da.types(k) == da.horizontalFlip
                            x(:,:, :, idx) = dp.imgFlip(x(:,:, :, idx), 2, da.p);
                        end
                        if da.types(k) == da.randomCrop
                            x(:, :, :, idx) = dp.randomCrop(x(:,:, :, idx), da.randomCropPad, da.randomCropSize, da.p);
                        end                       
                        if da.types(k) == da.verticalFlip
                            x(:,:, :, idx) = dp.imgFlip(x(:,:, :, idx), 1, da.p);
                        end
                        if da.types(k) == da.cutout
                            x(:,:, :, idx) = dp.cutout(x(:,:, :, idx), da.cutoutSize, da.p);
                        end
                    end
                end
            end
            x = reshape(x, [numel(x), 1]);
        end
        function [xcell, labelcell, labelStat] = regroupClasses(x, labels)
            minC = min(labels);
            maxC = max(labels);
            numClasses = maxC - minC + 1;
            xcell     = cell(numClasses, 1);
            labelcell = cell(numClasses, 1);
            labelStat = zeros(numClasses, 1);
            for c = 1:numClasses
                idx = labels==c-1;
                labelStat(c) = sum(idx);
                xcell{c}     = x(:,:,:,idx);
                labelcell{c} = labels(idx);
            end
        end
        function [xs, ls] = selectSamples(x, labels, numSamples)            
            [xcell, labelcell, labelStat] = dp.regroupClasses(x, labels);
            numClasses = size(xcell, 1);
            xs = cell(numClasses, 1);
            ls = cell(numClasses, 1);
            for c = 1:numClasses
               idx = randperm(labelStat(c), numSamples); 
               xs{c} = xcell{c}(:,:,:,idx);
               ls{c} = labelcell{c}(idx);
            end
            xs = cat(4, xs{:});
            ls = cat(1, ls{:});
        end
        % Data Augmentation
        function x = imgFlip(x, dim, p)
            a = binornd(1, p); 
            if dim == 1 && a == 1
                x = x(end:-1:1,:, :, :);
            elseif dim == 2 && a == 1
                x = x(:,end:-1:1,:, :);
            end
        end
        function x = randomCrop(x, pad, cropSize, p)
            wout = cropSize(1);
            hout = cropSize(2);
            a    = binornd(1, p); 
            [w, h, d, n] = size(x);
            if a == 1
                paddingImg = zeros(w+2*pad, h+2*pad, d, n, class(x));
                paddingImg(pad+1:end-pad, pad+1:end-pad, :, :) = x;
                idxW = randi(w+2*pad-wout+1)+(0:wout-1);
                idxH = randi(h+2*pad-hout+1)+(0:hout-1);
                x    = paddingImg(idxW, idxH, :, :);
            end
        end
        function x = cutout(x, rect, p)           
            a = binornd(1, p); 
            [w, h, d, n] = size(x); 
            if a == 1
                wcut = rect(1);
                hcut = rect(2);
                idxW = randi(w-wcut+1)+(0:wcut-1);
                idxH = randi(h-hcut+1)+(0:hcut-1);
                x(idxW, idxH, :, :) = zeros(wcut, hcut, d, n, 'like', x);
            end
        end
        function x = cvr2img(x, imgSize)
            w = imgSize(1);
            h = imgSize(2);
            d = imgSize(3);
            n = size(x, 1); 
            x = reshape(reshape(x',[w*h*d*n, 1]), [w, h, d, n]);
%             x = permute(x, [2 1 3 4]);
        end
        % Normalization
        function x = normalizeRGB(x, m, s, padding)
            D    = size(x, 3);
            for i = 1:D
                x(:,:,i,:)= (x(:,:,i,:) -  m(i))./s(i);
            end
            if padding>0
                x = dp.addPadding(x, padding);
            end
        end
        function x = denormalizeImg(x, m, s)
            D    = size(x, 3);
            for i = 1:D
                x(:,:,i,:)= x(:,:,i,:).*s(i) + m(i);
            end
        end
        function [xntrain, yntrain, xntest, yntest, mxtrain, sxtrain, mytrain, sytrain] = normalize(xtrain, ytrain, xtest, ytest)
            mxtrain = nanmean(xtrain);
            sxtrain = nanstd(xtrain);
            idx     = sxtrain==0;
            sxtrain(idx) = 1;
            mytrain = nanmean(ytrain);
            sytrain = nanstd(ytrain);
            xntrain = (xtrain - mxtrain)./sxtrain;
            yntrain = (ytrain - mytrain)./sytrain;
            xntest  = (xtest - mxtrain)./sxtrain;
            yntest  = ytest;
        end
        % 
        function [m, s] = meanstd(x)           
            nObs = size(x, 4);
            D    = size(x, 3);
            H    = size(x, 2);
            W    = size(x, 1);
            x    = permute(x, [1 2 4 3]);
            x    = reshape(x, [nObs*H*W, D]);
            m    = mean(x);
            s    = std(x);
        end
        function xp = addPadding(x, padding)
            nObs = size(x, 4);
            D    = size(x, 3);
            H    = size(x, 2);
            W    = size(x, 1);
            xp   = zeros(H+padding, W+padding, D, nObs);
            for k = 1:nObs
                for i = 1:D
                    xp(1:W,1:H,i,k)= x(:,:,i,k);
                end
            end
        end             
        function [xtrain, ytrain, xtest, ytest] = split(x, y, ratio)
            numObs      = size(x, 1);
            idxobs      = 1:numObs;
%             idxobs      = randperm(numObs);
            idxTrainEnd = round(ratio*numObs);
            idxTrain    = idxobs(1:idxTrainEnd);
            idxTest     = idxobs((idxTrainEnd+1):numObs);
            xtrain      = x(idxTrain, :);
            ytrain      = y(idxTrain, :);
            xtest       = x(idxTest, :);
            ytest       = y(idxTest, :);
        end
        function [trainIdx, testIdx] = indexSplit(numObs, ratio, dtype)
           idx = randperm(numObs);
           trainIdxEnd = round(numObs*ratio);
           trainIdx = idx(1:trainIdxEnd)';
           testIdx  = idx(trainIdxEnd+1:end)';
           if strcmp(dtype, 'single')
               trainIdx = int32(trainIdx);
               testIdx = int32(testIdx);
           end
        end
        function [x, y, labels, encoderIdx] = selectData(x, y, labels, encoderIdx, idx)
            x = x(:,:,:,idx);
            if ~isempty(y)
                y = y(idx, :);
            else
                y = [];
            end
            if ~isempty(labels)
                labels = labels(idx, :);
            else
                labels = [];
            end
            if ~isempty(encoderIdx)
                encoderIdx = encoderIdx(idx, :);
            else
                encoderIdx = [];
            end
        end
        function foldIdx = kfolds(numObs, numFolds)
            numObsPerFold = round(numObs/(numFolds));
            idx           = 1:numObsPerFold:numObs;
            if ~ismember(numObs, idx)
                idx = [idx, numObs];
            end
            if length(idx)>numFolds+1
                idx(end-1) = []; 
            end
            foldIdx = cell(numFolds, 1);
            for i = 1:numFolds
                if i == numFolds
                    foldIdx{i} = [idx(i):idx(i+1)]';
                else
                    foldIdx{i} = [idx(i):idx(i+1)-1]';
                end
            end
        end       
        function [xtrain, xval] = regroup(x, foldIdx, valfold)
            trainfold       = 1:size(foldIdx, 1);
            trainfold(valfold) = [];
            xval            = x(foldIdx{valfold}, :);
            trainIdx        = cell2mat(foldIdx(trainfold));
            xtrain          = x(trainIdx, :);
        end
        function [y, sy] = denormalize(yn, syn, myntrain, syntrain)
            y   = yn.*syntrain + myntrain;
            if ~isempty(syn)
                sy  = (syntrain.^2).*syn;
            else
                sy  = [];
            end
        end
        function y  = transformObs(y)
            maxy    = 10;
            miny    = -10;
            idx     = logical(y);
            y(idx)  = maxy;
            y(~idx) = miny;
        end
        function prob  = probFromloglik(loglik)
            maxlogpdf = max(loglik);
            w_1       = bsxfun(@minus,loglik,maxlogpdf);
            w_2       = log(sum(exp(w_1)));
            w_3       = bsxfun(@minus,w_1,w_2);
            prob      = exp(w_3);
        end
        function [y, idx]   = encoder(yraw, numClasses, dtype)
            [~, idx_c]=dp.class_encoding(numClasses);
            y   = zeros(size(yraw, 1), max(max(idx_c)), dtype);
            if strcmp(dtype, 'single')
                idx = zeros(size(yraw, 1), size(idx_c, 2), 'int32');
            elseif strcmp(dtype, 'double')
                idx = zeros(size(yraw, 1), size(idx_c, 2), 'int64');
            end            
            for c = 1:numClasses
                idxClasses         = yraw==c-1;
                [idxLoop, obs]     = dp.class2obs(c, dtype, numClasses);
                y(idxClasses, idxLoop) = repmat(obs, [sum(idxClasses), 1]);
                idx(idxClasses, :) = repmat(idxLoop, [sum(idxClasses), 1]);
            end
        end
        function idx        = selectIndices(idx, batchSize, numClasses, dtype)
            if strcmp(dtype, 'single')
                numClasses = int32(numClasses);
                idx        = int32(idx);
            elseif strcmp(dtype, 'double')
                numClasses = int64(numClasses);
                idx        = int64(idx);
            end
            for b = 1:batchSize
                idx(b, :) = idx(b, :) + (b-1)*numClasses;
            end
            idx = reshape(idx', [size(idx, 1)*size(idx, 2), 1]);
        end
        function [obs, idx] = class_encoding(numClasses)
            H=ceil(log2(numClasses));
            C=fliplr(de2bi([0:numClasses-1],H));
            obs=(-1).^C;
            idx=ones(numClasses,H);
            C_sum=[zeros(1,H),numClasses];
            for h=H:-1:1
                C_sum(h)=ceil(C_sum(h+1)/2);
            end
            C_sum=cumsum(C_sum)+1;
            for i=1:numClasses
                for h=1:H-1
                    idx(i,h+1)=bi2de(fliplr(C(i,1:h)))+C_sum(h);
                end
            end
            max_idx=max(max(idx));
            unused_idx=setdiff(1:max_idx,idx);
            for j=1:length(unused_idx)
                idx_loop=(idx-j+1)>(unused_idx(j));
                idx(idx_loop)=idx(idx_loop)-1;
            end
        end
        function [obs, idx] = class_encoding_V2(numClasses)
            H=ceil(log2(numClasses));
            C=fliplr(de2bi(colon(0, numClasses-1),H));
            obs=(-1).^C;
            idx=ones(numClasses,H);
            C_sum=[zeros(1,H),2^H];
            for h=H:-1:1
                C_sum(h)=ceil(C_sum(h+1)/2);
            end
            C_sum=cumsum(C_sum)+1;
            for i=1:numClasses
                for h=1:H-1
                    idx(i,h+1)=bi2de(fliplr(C(i,1:h)))+C_sum(h);
                end
            end
        end
        function [idx, obs] = class2obs(class, dtype, numClasses)
            [obs_c, idx_c]=dp.class_encoding(numClasses);
            idx=idx_c(class,:);
            obs=obs_c(class,:);
            if strcmp(dtype, 'single')
                idx = int32(idx);
                obs = single(obs);
            elseif strcmp(dtype, 'half')
                idx = int32(idx);
                obs = half(obs);
            end
        end
        function p_class    = obs2class(mz, Sz, obs_c, idx_c)          
            alpha = 3;
            p_obs = normcdf(mz./sqrt((1/alpha)^2 + Sz), 0, 1);           
            p_class=prod(abs(p_obs(idx_c)-(obs_c==-1)), 2);
        end  
        function [ytest, denorm] = MinmaxSc(ytrain, ytest, a, b)
            co      = (b-a)/(max(ytrain) - min(ytrain));
            ytest   = co.*(ytest-min(ytrain)) + a;
            denorm.co = co;
            denorm.mi = min(ytrain);
            denorm.ma = max(ytrain);
            denorm.a  = a;
            denorm.b  = b;
        end
        function [ytest, Sytest] = deMinmaxSc(ytest, Sytest, denorm)
            ytest  = (ytest-denorm.a)./denorm.co + denorm.mi;
            Sytest = Sytest./(denorm.co)^2;
            
        end
        %
        function [yMA] = ma(y,w)
            yMA = zeros(size(y));
            for i = 1:size(y,1)
               if i<=w 
                   yMA(i,:) = nanmean(y(1:w,:));
               else
                   yMA(i,:) = nanmean(y(i-w+1:i,:));
               end
            end
        end
       
    end
end