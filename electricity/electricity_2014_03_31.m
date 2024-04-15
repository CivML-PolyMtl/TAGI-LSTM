%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         Electricity 1 week of training - 1 week of prediction
% Description:  Apply TAGI to predict a sine wave
% Authors:      Van-Dai Vuong, Luong-Ha Nguyen & James-A. Goulet
% Created:      Apr 22, 2022
% Updated:      Apr 22, 2022
% Contact:      vuongdai@gmail.com, luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2022 Van-Dai Vuong, Luong-Ha Nguyen & James-A. Goulet 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all

% initSeed = randi([1,1000],[5,1]);
initSeed = [361 506 551 262 17];

 
%% Data
path  = char([cd ,'/electricity']);
data  = load(char([path, '/electricity.mat']));
tsIdx = [1:370];
y     = data.elec(:,tsIdx);

nbTS  = numel(tsIdx);
% plot(y)
batchSize = 4;
nb_tsPar = 24; 
t1 = datenum('01-Jan-2012 00:00:00','dd-mmm-yyyy HH:MM:SS');
tend = t1 + 1/24*(size(y,1)-1);
t = [t1:1/24:tend]';
t = t(19513:19848); % time between 01-01-2014 to 07-09-2014
y = y(19513:19848,:);
% plot(y)
[~, ~, t_.day_month, t_.hour_day] = datevec(t);
[t_.day_week] = weekday(t);
x = [t_.hour_day,t_.day_week];
[x_norm, ~, ~, ~, ~, ~, ~, ~] = dp.normalize(x, [], x, []);
 
% Option:
nbobs    = size(y,1);        % total number of observations         
nbtest   = 7*24;             % number of test point
sql = 24;  
net.sql  = sql;             % Lookback period
net.xsql = 1;
net.nbCov = size(x,2);  % number of covariates
nbtrain  = nbobs-nbtest;     % number of training point
nbval    = 1*24;
RollWindow = 24;  % n-step-ahead predictions: test
Val_transfer_mem = 0;   % transfer c/h from val to test
Val_transfer_sq = 0;   % 1: use z^(o) 0: use observations for sequence
Test_transfer_mem = 0;   % transfer c/h from val to test
Test_transfer_sq = 0;   % 1: use z^(o) 0: use observations for sequence

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
net = rnn.defineNet(net,  sv_up,   batchSize,    maxEpoch,    [7 7 7],    [40 40 40]);
              %net  sv      batchSize   MaxEpoch    layer   node   
net.gainS = [.5,2,4,1].*ones(1, length(net.layer)-1);     
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
    yvalPd_ts  = cell(nbTsLoop,net.maxEpoch);
    SyvalPd_ts = cell(nbTsLoop,net.maxEpoch);
    memTest_ts  = cell(nbTsLoop,net.maxEpoch);
    ytestPd_ts  = cell(nbTsLoop,1);
    SytestPd_ts = cell(nbTsLoop,1);
    memVal  = cell(nbTsLoop,1);
    %
    for epoch = 1:maxEpoch
        % Re-initialization after running each epoch
        theta   = cell(nbTsLoop,1);
        LL      = nan(nbTsLoop,1);
        yvalPd  = cell(nbTsLoop,1);
        SyvalPd = cell(nbTsLoop,1);
        memTest = cell(nbTsLoop,1);
        net.sv   = svGrid(epoch);
        netT.sv   = svGrid(epoch);
        parfor j = 1:nbTsLoop
            % Train
            [xtrain_loop, ytrain_loop, nb_del] = tagi.prepDataBatch_RNN (xtrain, ytrain_ts(:,j), batchSize, sql);
            lstm = [];
            lstm.theta = theta_ts{j,epoch};
            lstm.Mem = Mem0;
            [ytrainPd, SytrainPd, theta{j}, memVal{j}] = task.runLSTM(net, lstm, xtrain_loop, ytrain_loop);

            %% Validation
            lstmT = [];
            lstmT.theta = theta{j};
            if Val_transfer_mem == 1 && Val_transfer_sq == 1
                lstmT.Mem = rnn.getMem (net.layer, memVal{j}, batchSize);
                lstmT.Sq  = rnn.getSq (sql, ytrainPd, SytrainPd);
            elseif Val_transfer_mem == 0 && Val_transfer_sq == 0
                lstmT.Mem = Mem0_valTest;
                lstmT.Sq  = rnn.getSq (sql, ytrain_loop, zeros(size(ytrain_loop)));
            elseif Val_transfer_mem == 1 && Val_transfer_sq == 0
                lstmT.Mem = rnn.getMem (net.layer, memVal{j}, batchSize);
                lstmT.Sq  = rnn.getSq (sql, ytrain_loop, zeros(size(ytrain_loop)));
            end
            [yvalPd_, SyvalPd_, memTest{j}, yvalPd{j}, SyvalPd{j}] = rnn.lstmTest(netT, lstmT, xval, yval_ts(:,j), yvalnomask_ts(:,j), RollWindow, Val_transfer_sq);

%             figure(1)
%             pl.plPrediction (t(valIdx)', yval_loop_nomask(end-nbval+1:end,:), t(valIdx)', yvalPd_temp, SyvalPd_temp, epoch)
%             xlim ([t(valIdx(1)) t(valIdx(end))]);
%             pause(0.01)
%             clf('reset')

            % LLval
            [yvalPd_, SyvalPd_] = dp.denormalize(yvalPd_, SyvalPd_, mytrain_ts(j), sytrain_ts(j));
            LL(j)  = gather(mt.loglik(y_ts(valIdx,j), yvalPd_, SyvalPd_));
            disp(['TS #' num2str(idxTs(j)) '. Epoch ' num2str(epoch) '/' num2str(maxEpoch) '. LL :' num2str(LL(j))])
        end
        theta_ts(:,epoch+1) = theta;
        yvalPd_ts(:,epoch)  = yvalPd;
        SyvalPd_ts(:,epoch) = SyvalPd;
        memTest_ts(:,epoch)  = memTest;
        LL_ts(:,epoch)      = LL;
    end
    
    % find optimal epoch of each batch of TS
    theta_ts = theta_ts(:,2:end);
    LL_ts(:,1:nbEpoch_decay) = nan;
    [~, idx_epochOP] = max(LL_ts,[],2,'omitnan'); 
    [~, idx_optim] = max(LL_ts,[],2,'omitnan','linear'); 
    epoch_optim(idxTs) = idx_epochOP;
    theta_ts   = theta_ts(idx_optim);  % check
    yvalPd_ts  = yvalPd_ts(idx_optim); % check
    SyvalPd_ts = SyvalPd_ts(idx_optim); % check
    memTest_ts  = memTest_ts(idx_optim); % check
    theta(idxTs)   = theta_ts;
    LL_val(idxTs)  = LL_ts(idx_optim);

    %% Test
    parfor j = 1:nbTsLoop
        lstmT = [];
        lstmT.theta = theta_ts{j};
        if Test_transfer_mem == 1 && Test_transfer_sq == 1
            lstmT.Mem = memTest_ts{j};
            lstmT.Sq  = rnn.getSq (sql, yvalPd_ts{j}, SyvalPd_ts{j});
        elseif Test_transfer_mem == 0 && Test_transfer_sq == 0
            lstmT.Mem = Mem0_valTest;
            Sq_ = [ytrain_ts(:,j); yvalnomask_ts(:,j)];
            lstmT.Sq  = rnn.getSq (sql, Sq_, zeros(size(Sq_)));
        elseif Test_transfer_mem == 1 && Test_transfer_sq == 0
            lstmT.Mem = memTest_ts{j};
            Sq_ = [ytrain_ts(:,j); yvalnomask_ts(:,j)];
            lstmT.Sq  = rnn.getSq (sql, Sq_, zeros(size(Sq_)));
        end
        [ytestPd_, SytestPd_] = rnn.lstmTest(netT, lstmT, xtest, ytest_ts(:,j), ytestnomask_ts(:,j), RollWindow, Test_transfer_sq);
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

% plot
ttrain = t(trainValIdx)';
tval   = t(valIdx)';
ttest  = t(testIdx)';
 
for j = 1:4
figure
pl.plPrediction (t([valIdx;testIdx]), y([valIdx;testIdx],j), ttest, ytestPd(:,j), SytestPd(:,j), []);
xlim ([t(valIdx(1)) t(end)]);
title(['TS #' num2str(tsIdx(j))])
end   
 
% disp(['Optimized ep: #' num2str(epoch_optim)]) 
ND = mt.computeND(y(testIdx,:),ytestPd)
LL = mt.loglik(y(testIdx,:), ytestPd(:,:), SytestPd(:,:))
QL = mt.compute90QL (y(testIdx,:), ytestPd(:,:), SytestPd(:,:))

% save('elec_1week_ensemSv05_verify','y','t','ytestPd','SytestPd','initSeed','epoch_optim','testIdx','tsIdx','ytestPd_cell','SytestPd_cell','net','netT')
 
 
 
 
 
 
 
 
