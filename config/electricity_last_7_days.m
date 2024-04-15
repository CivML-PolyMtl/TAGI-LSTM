%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         electricity_last_7_days
% Description:  electricity_last_7_days
% Authors:      Van-Dai Vuong, Luong-Ha Nguyen & James-A. Goulet
% Created:      Apr 22, 2022
% Updated:      Apr 15, 2024
% Contact:      vuongdai@gmail.com, luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2020 Van-Dai Vuong, Luong-Ha Nguyen & James-A. Goulet 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all
 
% initSeed = randi([1,1000],[5,1]);
initSeed =[976 160 301 600 474];
 
%% Data
path  = char([cd ,'/data']);
data  = load(char([path, '/electricity.mat']));
tsIdx = [1:370];
y     = data.elec(:,tsIdx);
nbTS  = numel(tsIdx);

batchSize = 16;
nb_tsPar = 24;    % number of time series running in parallel using parfor function

% time covariates
time_start = datenum('01-Jan-2012 00:00:00','dd-mmm-yyyy HH:MM:SS');
time_end = time_start + 1/24*(size(y,1)-1);
t = [time_start:1/24:time_end]';
[~, ~, t_.day_month, t_.hour_day] = datevec(t);
[t_.day_week] = weekday(t);
x = [t_.hour_day,t_.day_week];
[x_norm, ~, ~, ~, ~, ~, ~, ~] = dp.normalize(x, [], x, []);
 
% Option:
nbobs    = size(y,1);        % total number of observations         
nbtest   = 7*24;             % number of test point
sql = 7*24;  
net.sql  = sql;              % Lookback period
net.xsql = 1;
net.nbCov = size(x,2);       % number of covariates
nbtrain  = nbobs-nbtest;     % number of training point
nbval    = 14*24;
RollWindow = 24;             % n-step-ahead predictions for rolling window predictions

%% Data split and normalization
[mytrain, sytrain, ytrain, yval, yval_nomask,...
ytest, ytest_nomask, xtrain, xval, xtest, trainIdx, valIdx, ...
trainValIdx, testIdx] = rnn.RnnDataProcess (x_norm, y, nbobs, nbval, nbtest);
 
ytestPd_cell  = cell(length(initSeed),1);
SytestPd_cell = cell(length(initSeed),1);
for i = 1:length(initSeed)
close all
set(0,'DefaultAxesFontName','Helvetica')
set(0,'defaultLineLineWidth',2)
set(0,'DefaultAxesFontSize',14)
set(0,'defaulttextfontsize',14)
format shortE
rng(initSeed(i))
 
%% Network
sv_up = 1;
sv_low = 0.5;
nbEpoch_decay = 5;
maxEpoch = 50;
[svGrid] = rnn.svgrid (sv_up, sv_low, nbEpoch_decay, maxEpoch);
net = rnn.defineNet(net,  sv_up,   batchSize,    maxEpoch,    [7 7 7],    [40 40 40]);    % 7 is TAGI-LSTM layer
                    %net  sv       batchSize     MaxEpoch     layer        node   
net.gainS = [.5,2,2,1].*ones(1, length(net.layer)-1);     
netT = net;     
net.trainMode = 1;
net.batchSize = batchSize;
netT.trainMode = 0;
netT.batchSize = 1;
[net, states, maxIdx, netInfo] = network.initialization(net);        
theta   = cell(nbTS,1);
LL_val  = nan(nbTS,1);
epoch_optim = zeros(nbTS,1);
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
Mem0_valTest = rnn.initializeRnnMemory_v1 (net.layer, net.nodes, netT.batchSize, m_Mem, S_Mem);
 
%% Run
disp(['Seed  : #' num2str(i)])
disp(['Training................'])
for ts = 1:nb_tsPar:nbTS
    % Re-initialization after running each batch of TS in parfor-loop
    idxTs = ts:1:min(ts+nb_tsPar-1,nbTS);
    nbTsLoop = numel(idxTs);
    ytrain_ts = ytrain(:,idxTs);
    yval_ts  = yval(:,idxTs);
    yvalnomask_ts  = yval_nomask(:,idxTs);
    ytest_ts = ytest(:,idxTs);
    ytestnomask_ts  = ytest_nomask(:,idxTs);
    mytrain_ts = mytrain(idxTs);
    sytrain_ts = sytrain(idxTs);
    y_ts  = y(:,idxTs); 
    theta_ts = cell(nbTsLoop,net.maxEpoch+1);
    theta0 = cell(nbTsLoop,1);
    for k = 1:numel(idxTs)
        theta0{k,1} = tagi.initializeWeightBias(net);
    end
    theta_ts(:,1) = theta0;
    LL_ts = nan(nbTsLoop,net.maxEpoch);
    ytestPd_ts  = cell(nbTsLoop,1);
    SytestPd_ts = cell(nbTsLoop,1);
    %
    for epoch = 1:maxEpoch
        % Re-initialization after running each epoch
        theta   = cell(nbTsLoop,1);
        LL      = nan(nbTsLoop,1);
        net.sv   = svGrid(epoch);
        netT.sv   = svGrid(epoch);
        parfor j = 1:nbTsLoop
            % Train
            [xtrain_loop, ytrain_loop, nb_del] = tagi.prepDataBatch_RNN (xtrain, ytrain_ts(:,j), batchSize, sql);
            lstm = [];
            lstm.theta = theta_ts{j,epoch};
            lstm.Mem = Mem0;
            [ytrainPd, SytrainPd, theta{j}] = task.runLSTM(net, lstm, xtrain_loop, ytrain_loop);

            %% Validation
            lstmT = [];
            lstmT.theta = theta{j};
            lstmT.Mem = Mem0_valTest;
            lstmT.Sq  = rnn.getSq (sql, ytrain_loop, zeros(size(ytrain_loop)));
            [yvalPd_, SyvalPd_] = rnn.lstmTest(netT, lstmT, xval, yval_ts(:,j), yvalnomask_ts(:,j), RollWindow);

            % LLval
            [yvalPd_, SyvalPd_] = dp.denormalize(yvalPd_, SyvalPd_, mytrain_ts(j), sytrain_ts(j));
            LL(j)  = gather(mt.loglik(y_ts(valIdx,j), yvalPd_, SyvalPd_));
            disp(['TS #' num2str(idxTs(j)) '. Epoch ' num2str(epoch) '/' num2str(maxEpoch) '. LL :' num2str(LL(j))])
        end
        theta_ts(:,epoch+1) = theta;
        LL_ts(:,epoch)      = LL;
    end
    
    % find optimal epoch of each batch of TS
    theta_ts = theta_ts(:,2:end);
    LL_ts(:,1:nbEpoch_decay) = nan;
    [~, idx_epochOP] = max(LL_ts,[],2,'omitnan'); 
    [~, idx_optim] = max(LL_ts,[],2,'omitnan','linear'); 
    epoch_optim(idxTs) = idx_epochOP;
    theta_ts   = theta_ts(idx_optim);  

    %% Test
    parfor j = 1:nbTsLoop
        lstmT = [];
        lstmT.theta = theta_ts{j};
        lstmT.Mem = Mem0_valTest;
        Sq_ = [ytrain_ts(:,j); yvalnomask_ts(:,j)];
        lstmT.Sq  = rnn.getSq (sql, Sq_, zeros(size(Sq_)));
        [ytestPd_, SytestPd_] = rnn.lstmTest(netT, lstmT, xtest, ytest_ts(:,j), ytestnomask_ts(:,j), RollWindow);
        % LLtest
        [ytestPd_ts{j}, SytestPd_ts{j}] = dp.denormalize(ytestPd_, SytestPd_, mytrain_ts(j), sytrain_ts(j));
    end
    ytestPd(idxTs)  = ytestPd_ts;
    SytestPd(idxTs) = SytestPd_ts;
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
ND = mt.computeND(y(testIdx,:),ytestPd)
QL = mt.compute90QL (y(testIdx,:), ytestPd(:,:), SytestPd(:,:))
RMSE = mt.computeRMSE(y(testIdx,:), ytestPd)
 
% save results
% save('electricity_last_7_days','ytestPd','SytestPd','initSeed')
 
 
 
 
 
 

