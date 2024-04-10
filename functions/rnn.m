%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         Neural network
% Description:  Neural network for TAGI
% Authors:      Van-Dai Vuong & James-A. Goulet 
% Created:      Jan 06, 2022
% Updated:      Jan 06, 2022
% Contact:      van-dai.vuong@polymtl.ca, luongha.nguyen@gmail.com & james.goulet@polymtl.ca 
% Copyright (c) 2022 Van-Dai Vuong, Luong-Ha nguyen  & James-A. Goulet  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef rnn
    methods (Static)
        function net = defineNet(net, sv, batchSize, maxEpoch, layer, nodes)
            %% Network
            if layer(1) == 7
                net.RNNtype   = 'LSTM_lookback';    % 2 types of LSTM: 'LSTM_stateful' 'LSTM_lookback'
            elseif layer(1) == 8
                net.RNNtype   = 'GRU_lookback';    % GRU
            end
            net.RNNbias = 0;  
%             net.smoother = 0;
            net.teacherForcing = 'zO';         % 'zO': z_outpuy or 'obs': observation
            if ~isfield(net,'xsql')
                net.xsql = 0;
            end
            if ~isfield(net,'nbCov')
                net.nbCov = 0;
            end
            if ~isfield(net,'rollWindow')
                net.rollWindow = 1;
            end
            nx = net.sql + net.xsql*net.nbCov;
            ny = 1;
            % Observation noise
            if ~isfield(net,'learnNoise')
                net.learnSv        = 0;% Online noise learning
                net.sv             = sv*ones(1,1);
                net.noiseType      = [];
            elseif net.learnNoise == 1 
                ny = 2;
                net.sv = [];
            end
            % GPU 1: yes; 0: no
            net.task           = 'regression';
            net.modelName      = '';
            net.dataName       = '';
            net.cd             = cd;
            net.saveModel      = 0;
            % GPU
            net.gpu            = 0;
            net.numDevices     = 0;
            % Data type object half or double precision
            net.dtype          = 'single';
            % Number of input covariates
            net.nx             = nx;
            % Number of output responses
            net.nl             = ny;
            net.nv2            = ny;
            net.ny             = ny;
            % Batch size
            net.batchSize      = batchSize;
            net.repBatchSize   = 1;
            % Layer| 1: FC; 2:conv; 3: max pooling;
            % 4: avg pooling; 5: layerNorm; 6: batchNorm; 7:LSTM
            % Activation: 1:tanh; 2:sigm; 3:cdf; 4:relu; 5:softplus
            net.layer          = [1,layer,1 ];
            net.nodes          = [nx, nodes, ny];
            net.actFunIdx      = [0, zeros(size(layer)),0];
            % LSTM activation functions for gates
            net.gateActFunIdx  = [2 2 1 2];
            % Parameter initialization
            net.initParamType  = 'He'; %'He' 'Xavier'
            net.gainS          = 1*ones(1, length(net.layer)-1);
            % Maximal number of epochs and splits
            net.maxEpoch       = maxEpoch;
            net.numSplits      = 1;  % for LSTM, RNN: net.numSplits = 1, no data permutation
       end
        function [xSq, ySq] = RnnSequence (x, y, sql)
            % Prepare sequence data for training: xSq and ySq   
            numSq = size(y,1)-sql;
            nb_Cov = size(x,2);
            xSq = nan(numSq, sql+nb_Cov);
            ySq = nan(numSq,1);
            x(1:sql,:) = []; 
            y = y';
            for j = 1:1:numSq
                xSq(j,:) = [x(j,:), y(j:j+sql-1)] ;
                ySq(j,:) = y(j+sql);
            end
        end
        function [xtest, ytest, ytest_nomask] = RollWindowData (xtest, ytest, RollWindow, nbAdd_ValTest)
            nbWindow = ceil((size(ytest,1)-nbAdd_ValTest)/RollWindow);
            a = min(RollWindow,size(ytest,1)-nbAdd_ValTest);
            ytest_ = nan(nbAdd_ValTest+a,size(ytest,2),nbWindow);
            ytest_nomask = nan(nbAdd_ValTest+a,size(ytest,2),nbWindow);
            xtest_ = nan(nbAdd_ValTest+a,size(xtest,2),nbWindow);
            for i = 1:nbWindow
                idxStart = (i-1)*RollWindow+1;
                b = size(ytest,1) - nbAdd_ValTest - idxStart+1;
                if b>=a
                    idxEnd   = idxStart+nbAdd_ValTest+a-1;
                    ytemp    = ytest(idxStart:idxEnd,:);
                    ytest_nomask(:,:,i) = ytemp;
                    ytemp(end-a+1:end,:) = nan;
                    ytest_(:,:,i) = ytemp;
                    xtest_(:,:,i) = xtest(idxStart:idxEnd,:);
                else
                    idxEnd   = idxStart+nbAdd_ValTest+b-1;
                    ytest_(1:length(idxStart:idxEnd),:,i) = ytest(idxStart:idxEnd,:);
                     ytest_nomask(:,:,i) = ytemp;
                    ytest_(end-a+1:end,:,i) = nan;
                    xtest_(1:length(idxStart:idxEnd),:,i) = xtest(idxStart:idxEnd,:);
                end
            end
            ytest = ytest_;
            xtest = xtest_;
        end
        function day_of_week = dayOfWeek_mat2py(day_of_week)
            % matlab day of week: sun=1 -> sar=7
            idx_mon = find(day_of_week==2);
            idx_tue = find(day_of_week==3);
            idx_wed = find(day_of_week==4);
            idx_thu = find(day_of_week==5);
            idx_fri = find(day_of_week==6);
            idx_sar = find(day_of_week==7);
            idx_sun = find(day_of_week==1);
            % convert day of week equivalent to pandas: mon = 0 -> sun=6
            day_of_week(idx_mon) = 0;
            day_of_week(idx_tue) = 1;
            day_of_week(idx_wed) = 2;
            day_of_week(idx_thu) = 3;
            day_of_week(idx_fri) = 4;
            day_of_week(idx_sar) = 5;
            day_of_week(idx_sun) = 6;
        end
        function [rnnMem] = initializeRnnMemory (layer, node, batchSize, init_value)
            % LSTM
            nblayer = numel(layer);
            rnnMem_  = cell(nblayer,1);
            for i = 1:nblayer
                if layer(i) == 7 ||  layer(i) == 8
                    rnnMem_{i} = init_value.*ones(node(i)*batchSize, 1, 'single');
                else
                    rnnMem_{i} = [];
                end
            end
            if any(layer == 7)
                rnnMem(1:4,1) = {rnnMem_}; % LSTM
            elseif any(layer == 8)
                rnnMem(1:2,1) = {rnnMem_}; % GRU
            end
        end
        function [rnnMem] = initializeRnnMemory_v1 (layer, node, batchSize, init_m, init_v)
            % LSTM
            nblayer = numel(layer);
            rnnMem_m  = cell(nblayer,1);
            rnnMem_v  = cell(nblayer,1);
            for i = 1:nblayer
                if layer(i) == 7 ||  layer(i) == 8
%                     rnnMem_m{i} = init_m.*ones(node(i)*batchSize, 1, 'single');
                    rnnMem_m{i} = randn(node(i)*batchSize, 1,'single').*sqrt(init_v);
                    rnnMem_v{i} = init_v.*ones(node(i)*batchSize, 1, 'single');
                else
                    rnnMem_{i} = [];
                end
            end
            if any(layer == 7)
                rnnMem([1,3],1) = {rnnMem_m}; % LSTM
                rnnMem([2,4],1) = {rnnMem_v}; % LSTM
            elseif any(layer == 8)
                rnnMem(1:2,1) = {rnnMem_}; % GRU
            end
        end
        function [rnnMem_merge] = RnnMergeMemory4batch (rnnMem)
            rnnMem_merge = cell(size(rnnMem,1),1);
            for i = 1:size(rnnMem,1)
                mem = cat(2,rnnMem{i,:});
                mem_ = cell(size(mem,1),1);
                for j=1:size(mem,1)
                    mem_{j} = vertcat(mem{j,:});
                end
                rnnMem_merge{i} = mem_;
            end
        end
        function [rnnMem_split] = RnnSplitMemoryBatch1(layer, rnnMem, batchSize)
            rnnMem_split_ = cell(numel(layer), batchSize);
            rnnMem_split = cell(4,batchSize);
            for i = 1:size(rnnMem,1)
                mem_ = rnnMem{i};
                for j = 1:size(mem_,1)
                    if layer(j) == 7 || layer(j) == 8
                        mem = reshape(mem_{j},[],batchSize);
                        node = size(mem,1);
                        mem = mat2cell(mem,node,ones(1,batchSize));
                    else
                        mem = cell(1,batchSize);
                    end
                    rnnMem_split_(j,:) = mem;
                end
                for k = 1:batchSize
                    rnnMem_split(i,k) = {rnnMem_split_(:,k)};
                end
            end
        end
        function parameter = initializeOrthogonal(sz)
            Z = randn(sz,'single');
            [Q,R] = qr(Z,0);
            D = diag(R);
            Q = Q * diag(D ./ abs(D));
            parameter = single(Q);
        end
        function [mytrain, sytrain, ytrain, yval, yval_nomask,...
                ytest, ytest_nomask, xtrain, xval, xtest, trainIdx, valIdx, ...
                trainValIdx, testIdx, xTrainVal, yTrainVal] = RnnDataProcess (x_norm, y, nbobs, nbval, nbtest)
            trainIdx    = [1:1:nbobs-nbtest-nbval]';               % traning indices
            valIdx      = [nbobs-nbtest-nbval+1:1:nbobs-nbtest]';  % validation indices
            trainValIdx = [trainIdx;valIdx];                    % training & validation indices
            testIdx     = [nbobs-nbtest+1:1:nbobs]';            % test indices
            
            % Train and Val
            [~, ~, yTrainVal, ~, ~, ~, mytrain, sytrain] = dp.normalize(y(trainValIdx,:), y(trainValIdx,:), y(trainValIdx,:), []);
            ytrain = yTrainVal(trainIdx,:);
            xtrain = x_norm(trainIdx,:,:);
            yval_nomask   = yTrainVal(valIdx,:);
            yval = nan(size(yval_nomask));
            xval   = x_norm(valIdx,:);
            xTrainVal = x_norm(trainValIdx,:,:);
            
            % Testing set
            xtest = x_norm(testIdx,:);
            [~, ~, ytest_nomask, ~, ~, ~, ~, ~] = dp.normalize(y(trainValIdx,:), y(trainValIdx,:), y(testIdx,:), []);
            ytest = nan(size(ytest_nomask));
        end

        function [svGrid] = svgrid (sv_up, sv_low, nbEpoch_decay, maxEpoch)
            decay_factor = nthroot(sv_low/sv_up,nbEpoch_decay);
            svGrid = [];
            for ep=0:nbEpoch_decay-1
                svGrid =  [svGrid, sv_up*decay_factor^ep];
            end
            svGrid = [svGrid, sv_low.*ones(1,maxEpoch-nbEpoch_decay)];
        end

        function Sq = getSq (sql, y, Sy)
            y  = y(:);
            Sy = Sy(:);
            Sq{1,1} = y(end-sql+1:end,:);
            Sq{2,1} = Sy(end-sql+1:end,:);
        end
        function mem = getMem (layer, mem, batchSize)
            rnnMem_split = rnn.RnnSplitMemoryBatch1(layer, mem, batchSize);
            mem = rnnMem_split(:,end);
        end
        function [yPd, SyPd, mem, yPd_pos, SyPd_pos] = lstmTest(net, lstm, x, y, y_nomask, RollWindow, transfer_sq)
            yPd_pos  = lstm.Sq{1};
            SyPd_pos = lstm.Sq{2};
            if isnan(RollWindow)
                 [yPd, SyPd, ~, mem] =task.runLSTM(net, lstm, x, y);
            else
                mem0 = lstm.Mem;
                Sq0  = lstm.Sq;
                nbWindow = ceil(size(y,1)/RollWindow);
                yPd_prior  = cell(nbWindow,1);
                SyPd_prior = cell(nbWindow,1);
                for i = 1:nbWindow 
                    if transfer_sq == 1
                        lstm.Sq = rnn.getSq (net.sql, yPd_pos, SyPd_pos - net.sv^2);
                    elseif transfer_sq == 0
                        lstm.Sq = rnn.getSq (net.sql, yPd_pos, zeros(size(yPd_pos)));
                    end
                    xloop = x((i-1)*RollWindow+1:min(i*RollWindow,size(y,1)),:,:);
                    yloop = y((i-1)*RollWindow+1:min(i*RollWindow,size(y,1)),:);
                    [xloop, yloop] = tagi.prepDataBatch_RNN (xloop, yloop, 1, net.sql);
                    [yPd_prior{i}, SyPd_prior{i},~, ~] = task.runLSTM(net, lstm, xloop, yloop);
                    yloop = y_nomask((i-1)*RollWindow+1:min(i*RollWindow,size(y,1)),:);
                    [yPd_, SyPd_, ~, mem] = task.runLSTM(net, lstm, xloop, yloop);
                    if transfer_sq == 1
                        yPd_pos  = [yPd_pos ;yPd_];
                        SyPd_pos = [SyPd_pos;SyPd_];
                    elseif transfer_sq == 0
                        yPd_pos  = [yPd_pos ;yloop];
                        SyPd_pos = [SyPd_pos;zeros(size(yloop))];
                    end
                    lstm.Mem = mem;
                end
                yPd  = cell2mat(yPd_prior);
                SyPd = cell2mat(SyPd_prior);
                yPd_pos  = yPd_pos(net.sql+1:end,:);
                SyPd_pos = SyPd_pos(net.sql+1:end,:);
                if any(isnan(yPd)) || any(isnan(SyPd))
                    lstm.Sq  = Sq0;
                    lstm.Mem = mem0;
                    [yPd, SyPd, ~, mem] =task.runLSTM(net, lstm, x, y);
                end
            end
        end

    end
end
