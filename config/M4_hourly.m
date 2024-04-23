%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         M4 hourly
% Description:  M4 hourly
% Authors:      Van-Dai Vuong, Luong-Ha Nguyen & James-A. Goulet
% Created:      Aug 16, 2022
% Updated:      Apr 15, 2024
% Contact:      vuongdai@gmail.com, luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2022 Van-Dai Vuong, Luong-Ha Nguyen & James-A. Goulet 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all

initSeed = randi([1,1000],[3,1]);
% initSeed = 100;

%% Data
path  = char([cd ,'/data']);
data  = load(char([path, '/M4_hourly_timeCov.mat']));

tsIdx = [1:size(data.values,2)];
y = data.values;
x1 = data.hour_of_day;
x2 = data.day_of_week;
nbTS  = numel(tsIdx);

batchSize = 1;
nb_tsPar = 4;

ytestPd_cell  = cell(length(initSeed),1);
SytestPd_cell = cell(length(initSeed),1);

% time covariates
[x1_norm, ~, ~, ~, ~, ~, ~, ~] = dp.normalize(x1, [], x1, []);
[x2_norm, ~, ~, ~, ~, ~, ~, ~] = dp.normalize(x2, [], x2, []);

% Option:
nbtest   = 48;    % number of test point
nbval    = 12;
nbobs    = size(y,1);        % total number of observations
sql = 24*7;
net.sql  = sql;             % Lookback period
net.xsql = 1;
net.nbCov = 2;  % number of covariates
nbtrain  = nbobs-nbtest;     % number of training point
seasonality = 24*7;

%% Data split and normalization
[mytrain, sytrain, ytrain, yval, yval_nomask,...
    ytest, ytest_nomask, xtrain1, xval1, xtest1, trainIdx, valIdx, ...
    trainValIdx, testIdx, xTrainVal1, yTrainVal] = rnn.RnnDataProcess (x1_norm, y, nbobs, nbval, nbtest);
[~, ~, ~, ~, ~,...
    ~, ~, xtrain2, xval2, xtest2, ~, ~, ...
    ~, ~, xTrainVal2] = rnn.RnnDataProcess (x2_norm, y, nbobs, nbval, nbtest);

for i = 1:length(initSeed)
close all
set(0,'DefaultAxesFontName','Helvetica')
set(0,'defaultLineLineWidth',2)
set(0,'DefaultAxesFontSize',14)
set(0,'defaulttextfontsize',14)
format shortE
rng(initSeed(i))

%% BDLM
maxEpoch = 50;
comp = [113,7]; % 113: Local level + trend + exponential smoothing; 7:LSTM
sQ    = [0,0];
sv_Stability = 1E-5;
%% Network
net.learnNoise = 1;
net.noiseType = 'hete';
net = rnn.defineNet(net,  [],   batchSize,    maxEpoch,    [7],    [50]);    % 7 is TAGI-LSTM layer
                    %net  sv   batchSize     MaxEpoch     layer   node
net.lastLayerUpdate = 0; % update z^{o} by smoother equation (BDLM + LSTM)
netT = net;
net.trainMode = 1;
net.batchSize = batchSize;
netT.trainMode = 0;
netT.batchSize = 1;
netV = netT;
netV.trainMode = 2;
[net, states, maxIdx, netInfo] = network.initialization(net);
ytestPd  = cell(nbTS,1);  
SytestPd = cell(nbTS,1); 

% Initialize memory for LSTM: cell and hidden states
m_Mem = 0;    % initialized values for mh and mc (cell and hidden states)
S_Mem = 0;    % initialized values for Sh and Sc (cell and hidden states)
% Initialize LSTM's memory at t=0, epoch=1
% Mem{1} = mh (means for hidden states)
% Mem{2} = Sh (variances for hidden states)
% Mem{3} = mc (means for cell states)
% Mem{4} = Sc (variances for cell states)
Mem0 = rnn.initializeRnnMemory_v1 (net.layer, net.nodes, net.batchSize, m_Mem, S_Mem);

%% Run
disp(['Seed  : #' num2str(i)])
disp(['Training................'])

for ts = 1:nb_tsPar:nbTS
    idxTs = ts:1:min(ts+nb_tsPar-1,nbTS);
    nbTsLoop = numel(idxTs);
    theta  = cell(nbTsLoop,1);
    initx  = cell(nbTsLoop,1);
    initSx = cell(nbTsLoop,1);
    for j = 1:numel(idxTs)
        theta{j,1} = tagi.initializeWeightBias(net);
        y_temp = ytrain(:,idxTs(j));
        y_temp(isnan(y_temp)) = [];
%         initx{j,1}  = [nanmean(y_temp(1:seasonality)); 1E-2; 0; 0.4; 0; 0; 0];
%         initSx{j,1} = diag([1E-1; 1E-1; 0; 1E-2; 1E-10; 0; 0]);
        initx{j,1}  = [nanmean(y_temp(1:seasonality)); 1E-3; 1E-10; 1E-10; -3; 1E-5; 1E-5; 1E-1];
        initSx{j,1} = diag([1E-2; 1E-4; 1E-10; 1E-10; 1E-2; 1E-3; 1E-3; 1E-1]);
    end
    ytestPd_loop  = cell(nbTsLoop, maxEpoch);
    SytestPd_loop = cell(nbTsLoop, maxEpoch);
    metric_ts = nan(nbTsLoop, maxEpoch);

    for epoch = 1:maxEpoch
        parfor j = 1: nbTsLoop
            %% Train
            xtrain_loop  = [];
            xtrain_loop(:,1) = xtrain1(:,idxTs(j));
            xtrain_loop(:,2) = xtrain2(:,idxTs(j));
            ytrain_loop = ytrain(:,idxTs(j));
            idx_nan = isnan(ytrain_loop);
            idx = ~isnan(ytrain_loop);
            ytrain_loop(idx_nan) = []; 
            xtrain_loop(idx_nan,:) = []; 
            %

            [xtrain_loop, ytrain_loop] = tagi.prepDataBatch_RNN (xtrain_loop, ytrain_loop, batchSize, sql);
            lstm = [];
            lstm.theta = theta{j};
            lstm.Sq{1,1}  = 0*ones(sql,1);
            lstm.Sq{2,1}  = 0*ones(sql,1);
            lstm.Mem = Mem0;
            bdlm = [];
            bdlm.comp = comp;
            bdlm = BDLM.build(bdlm, sQ, sv_Stability, initx{j}, initSx{j});

            [~, ~, theta{j}, memTrain, xBu_train, SxBu_train, xBp_train, SxBp_train,~,~,Czz_train] = task.runHydrid_AGVI(net, lstm, bdlm, xtrain_loop, ytrain_loop);

            %% Validation
            %
            xval_loop = [];
            xval_loop(:,1) = xval1(:,idxTs(j));
            xval_loop(:,2) = xval2(:,idxTs(j));
            yval_loop = yval(:,idxTs(j));
            yval_nomask_loop = yval_nomask(:,idxTs(j));
            %
            lstmT = [];
            lstmT.theta = theta{j};
            lstmT.Mem = memTrain;
            lstmT.Sq  = rnn.getSq (sql, xBu_train(end,:), SxBu_train(end,:));
            bdlmT = [];
            bdlmT.comp = comp;
            bdlmT = BDLM.build(bdlmT, sQ, sv_Stability, xBu_train(:,end), SxBu_train(:,end));
            [yvalPd_, SyvalPd_] = task.runHydrid_AGVI(netT, lstmT, bdlmT, xval_loop, yval_loop);
            [~, ~, ~, memVal, xBu_val, SxBu_val, xBp_val, SxBp_val,~,~,Czz_val] = task.runHydrid_AGVI(netV, lstmT, bdlmT, xval_loop, yval_nomask_loop);
            [yvalPd, SyvalPd] = dp.denormalize(yvalPd_, SyvalPd_, mytrain(idxTs(j)), sytrain(idxTs(j)));
            metric = gather(mt.loglik(y(valIdx,idxTs(j)), yvalPd, SyvalPd));
            metric_ts(j,epoch)  = metric;

            disp(['TS #' num2str(idxTs(j)) '. Epoch ' num2str(epoch) '/' num2str(maxEpoch) '. Metric :' num2str(metric)])

            xBu  = [xBu_train xBu_val];
            SxBu = [SxBu_train SxBu_val];
            xBp  = [xBp_train xBp_val];
            SxBp = [SxBp_train SxBp_val];
            Czz = [Czz_train;Czz_val];

            [xBu, SxBu] = BDLM.KFSmoother_ESM_BNI(comp, xBp, SxBp, xBu, SxBu, bdlm.A, Czz);
            initx_ = xBu(:,1); initx_(1) = nanmean(ytrain_loop(1:seasonality)); initx_(3) = 0;
            initx{j}    = initx_;
            initSx{j}   = diag(diag(reshape(SxBu(:,1),size(xBu,1),[])));

            %% Test
            %
            xtest_loop = [];
            xtest_loop(:,1) = xtest1(:,idxTs(j));
            xtest_loop(:,2) = xtest2(:,idxTs(j));
            ytest_loop = ytest(:,idxTs(j));
            %
            lstmT.Mem = memVal;
            lstmT.Sq  = rnn.getSq (sql, xBu(end,:), SxBu(end,:));
            bdlmT = BDLM.build(bdlmT, sQ, sv_Stability, xBu(:,end), SxBu(:,end));
            [ytestPd_, SytestPd_,~,~,xBu_test, SxBu_test] = task.runHydrid_AGVI(netT, lstmT, bdlmT, xtest_loop, ytest_loop);
            [ytestPd_, SytestPd_] = dp.denormalize(ytestPd_, SytestPd_, mytrain(idxTs(j)), sytrain(idxTs(j)));
            %%  save
            ytestPd_loop(j,epoch)  = {ytestPd_};
            SytestPd_loop(j,epoch) = {SytestPd_};
        end
    end
    %% Optimal epoch
    [~, idx_epochOP] = max(metric_ts,[],2,'omitnan');
    [~, idx_optim]   = max(metric_ts,[],2,'omitnan','linear');
    ytestPd(idxTs)   = ytestPd_loop(idx_optim);
    SytestPd(idxTs)  = SytestPd_loop(idx_optim);
end
ytestPd  = reshape(ytestPd,1,[]);
ytestPd  = cell2mat(ytestPd);
SytestPd = reshape(SytestPd,1,[]);
SytestPd = cell2mat(SytestPd);
ytestPd_cell{i}  = ytestPd;
SytestPd_cell{i} = SytestPd;
end

nbmodel = length(initSeed);
[ytestPd, SytestPd] = ensem.ensembleWeight_cell(ytestPd_cell, SytestPd_cell, 1/nbmodel*ones(1,nbmodel));

% Display results
p50 = mt.computeND(y(end-nbtest+1:end,:),ytestPd)
p90 = mt.compute90QL(y(end-nbtest+1:end,:), ytestPd, SytestPd)
RMSE = mt.computeRMSE(y(end-nbtest+1:end,:), ytestPd)

% save('M4_hourly','ytestPd','SytestPd','initSeed')


 
 
 
 
 
 

