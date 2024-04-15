%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         Decompose TS #30 tourism monthly dataset using exponential smoothing component
% Description:  Analyse trended data using hyrbid model (SSM + LSTM) with
% exponential smoothing component
% Authors:      Van-Dai Vuong, Luong-Ha Nguyen & James-A. Goulet
% Contact:      vuongdai@gmail.com, luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2024 Van-Dai Vuong, Luong-Ha Nguyen & James-A. Goulet 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
close all
set(0,'DefaultAxesFontName','Helvetica')
set(0,'defaultLineLineWidth',2)
set(0,'DefaultAxesFontSize',14)
set(0,'defaulttextfontsize',14)
format shortE

%% #1. Data
initSeed = 1;       % Fixed seed to obtain reproductible results
rng(initSeed)

path  = char([cd '/data']);
data  = load(char([path, '/tourism_nanTop.mat']));
ts = 30;
y = data.month_values(:,ts);
x = data.month_timestamps(:,ts);
idxnan = isnan(y);
y(idxnan) = [];
x(idxnan) = [];
plot(y)

[x, ~, ~, ~, ~, ~, ~, ~] = dp.normalize(x, [], x, []);
 
%% 2. Option for model
nbobs    = size(y,1);        % total number of observations         
nbtest   = 24;             % number of test point
sql      = 12;  
net.sql  = sql;             % Lookback period
net.xsql = 1;
net.nbCov = size(x,2);  % number of covariates
nbtrain  = nbobs-nbtest;     % number of training point
nbval    = 12;

%% 3. Data split and normalizatio
[mytrain, sytrain, ytrain, yval, yval_nomask,...
ytest, ytest_nomask, xtrain, xval, xtest, trainIdx, valIdx, ...
trainValIdx, testIdx] = rnn.RnnDataProcess (x, y, nbobs, nbval, nbtest);

%% 4. State-space models (SSM)
bdlm.comp = [113,7]; % 113: Local level + trend + exponential smoothing; 7:LSTM
sQ    = [0,0];      % to construct transition noise matrix Q

sv_Stability = 1E-10; % to construct observation noise matrix R. Observtion equation: y=(x^L + x^E + x^LSTM + x^V) + v_Stability
% x^V: actual error is learnt as a hidden state using AGVI but need to add
% v_Stability ~ N(0,sv_Stability^2) for numerical stability 

% Hidden states for bdlm.comp = [113,7]
seasonality  = 12; % one cyle for data -> monthy data -> 12 (for initialization of hidden states only)
init_x=[
    % #1                                #2          #3             #4       #5                             #6              #7   
    % LL                                LT          Exp smoothing  alpha    sigma(alpha_{t-1})*v_{t-1}     Error(x^V)      LSTM 
	  nanmean(ytrain(1:seasonality));	1E-3;	 	0; 	           0.1;	    0;	                           0;	           0 
];

init_Sx=diag([
    % #1                                #2          #3             #4       #5                             #6              #7   
    % LL                                LT          Exp smoothing  alpha    sigma(alpha_{t-1})*v_{t-1}     Error(x^V)      LSTM 
	  1E-3;	                            1E-3;	 	0; 	           1E-1;	1E-10;	                       0;	           0 
]);

% Build bdlm: contain BDLM matrices, Q, V, hidden states
bdlm  = BDLM.build(bdlm, sQ, sv_Stability, init_x, init_Sx);

%% 5. Network
% 5.1 Initialize networks
net.learnNoise = 1;
batchSize = 1;
maxEpoch = 50;
sv = [];
% 7: LSTM; just define the number of LSTM layers and #LSTM node
net = rnn.defineNet(net,  sv,   batchSize,    maxEpoch,    [7],    [50]);
                    %net  sv      batchSize   MaxEpoch    layer   node
net.lastLayerUpdate = 0; % update z^{o} by smoother equation (BDLM + LSTM)
net.trainMode = 1;
net.batchSize = batchSize;
% Test network
netT = net; 
netT.trainMode = 0;
% Validation network
netT.batchSize = 1;
netV = netT;
netV.trainMode = 2;
[net, states, maxIdx, netInfo] = network.initialization(net);

% 5.2 Initialize parameters
theta = tagi.initializeWeightBias(net);

% 5.3 Initialize memory for LSTM: cell and hidden states
m_Mem = 1;    % initialized values for mh and mc (cell and hidden states)
S_Mem = 1;    % initialized values for Sh and Sc (cell and hidden states)
% Initialize LSTM's memory at t=0, epoch=1
% Mem{1} = mh (means for hidden states)
% Mem{2} = Sh (variances for hidden states)
% Mem{3} = mc (means for cell states)
% Mem{4} = Sc (variances for cell states)
Mem = rnn.initializeRnnMemory_v1 (net.layer, net.nodes, net.batchSize, m_Mem, S_Mem);

% 5.4 Initialize initial sequence length
m_Sq = 1.*ones(sql,1); % mean
S_Sq = 1.*ones(sql,1); % variances
% m_Sq = ytrain(1:sql); % mean
% S_Sq = 1.*ones(sql,1); % variances
Sq{1} = m_Sq; 
Sq{2} = S_Sq;

% 5.5 Initialize quantities
LL_optim = -1E100;
epoch_optim = 1;
%% 5. Analyze
disp(['Training................'])

for epoch = 1:maxEpoch
    % Train
    [xtrain_loop, ytrain_loop, nb_del] = tagi.prepDataBatch_RNN (xtrain, ytrain, batchSize, sql);
    % lstm initialization
    lstm       = [];
    lstm.theta = theta; 
    lstm.Sq    = Sq;
    lstm.Mem   = Mem;
    [~, ~, theta, memVal, xBu_train, SxBu_train, xBp_train, SxBp_train,~,~,Czz] = task.runHydrid_AGVI(net, lstm, bdlm, xtrain_loop, ytrain_loop);
    % Czz = cov(z^{O}_{t}, z^{O}_{t-1}) for smoothing

    % Validation
    lstmT = [];
    lstmT.theta = theta;
    lstmT.Mem = memVal;
    lstmT.Sq  = rnn.getSq (sql, xBu_train(end,:), SxBu_train(end,:));
    % bdlm val
    bdlmT = BDLM.build(bdlm, sQ, sv_Stability, xBu_train(:,end), SxBu_train(:,end)); %  build BDLM to again to have correct initial hidden states for valiation
    [yvalPd_, SyvalPd_] = task.runHydrid_AGVI(netT, lstmT, bdlmT, xval, yval); % first run to obtain multi-step prediction (prior predictive), yval = nan;
    [~, ~, ~, memTest, xBu_val, SxBu_val] = task.runHydrid_AGVI(netV, lstmT, bdlmT, xval, yval_nomask); % second run to obtain posteriors for hidden states, yval_nomask = real observations;
    [yvalPd, SyvalPd] = dp.denormalize(yvalPd_, SyvalPd_, mytrain, sytrain);
    LL = gather(mt.loglik(y(valIdx), yvalPd, SyvalPd));
    disp(['Epoch ' num2str(epoch) '/' num2str(maxEpoch) '. LL :' num2str(LL)])
    
    % Test
    lstmT = [];
    lstmT.theta = theta;
    lstmT.Mem = memTest;
    lstmT.Sq  = rnn.getSq (sql, [xBu_train(end,:), xBu_val(end,:)], [SxBu_train(end,:), SxBu_val(end,:)]);
    % bdlm test
    bdlmT = BDLM.build(bdlm, sQ, sv_Stability, xBu_val(:,end), SxBu_val(:,end)); %  build BDLM to again to have correct initial hidden states for test set
    [ytestPd_, SytestPd_,~,~, xBp_test, SxBp_test] = task.runHydrid_AGVI(netT, lstmT, bdlmT, xtest, ytest);
    [ytestPd_, SytestPd_]  = dp.denormalize(ytestPd_, SytestPd_, mytrain, sytrain);
    
    % Smoother
    [xBu_train, SxBu_train] = BDLM.KFSmoother_ESM_BNI(bdlm.comp, xBp_train, SxBp_train, xBu_train, SxBu_train, bdlm.A, Czz);
    init_x   = xBu_train(:,1);
    init_Sx  = reshape(SxBu_train(:,1),size(init_x,1),[]);
    init_Sx  = diag(diag(init_Sx));
    bdlm     = BDLM.build(bdlm, sQ, sv_Stability, init_x, init_Sx); %  build BDLM to again for the training set of next epoch

    % Save for optimal epoch
    if LL > LL_optim
        epoch_optim = epoch;
        LL_optim    = LL;
        ytestPd     = ytestPd_;
        SytestPd    = SytestPd_;
        x  = [xBu_train, xBu_val, xBp_test];
        Sx = [SxBu_train, SxBu_val, SxBp_test];
    end

    % % plot
    ttest = testIdx';
    tval  = valIdx';
    t = [trainIdx' valIdx' testIdx'];
    figure(1)
    subplot(4,1,1)
    pl.plPrediction (t, y, tval, yvalPd, SyvalPd, [],'r','k')
    ylabel('y')
    xlim([t(1) valIdx(end)])
    xline(trainIdx(end),'--')
    title ('Validation prediction')
    subplot(4,1,2)
    pl.plPrediction (t, [ytrain;yval_nomask;ytest_nomask], t, x(1,:)'+ x(3,:)', Sx(1,:)'+Sx(17,:)'+2*Sx(15,:)', [],'r','k')
    ylabel('x^L + x^E')
    xlim([t(1) valIdx(end)])
    xline(trainIdx(end),'--')
    title ('Level and exponential smoothing')
    subplot(4,1,3)
    pl.plPrediction ([], [], t, x(end,:)', Sx(end,:)', [],'r','k')
    ylabel('x^{LSTM}')
    xlim([t(1) valIdx(end)])
    xline(trainIdx(end),'--')
    title ('LSTM')
    subplot(4,1,4)
    plot( t, x(6,:)', LineWidth=0.8, Color='k')
%     pl.plPrediction ([], [], t, x(6,:)', Sx(41,:)', [],'r','k')
    ylabel('x^V')
    xlim([t(1) valIdx(end)])
    xline(trainIdx(end),'--')
    title ('Error hidden states')
    sgtitle(['Epoch: #' num2str(epoch)]) 
    pause(0.1)
    if epoch < maxEpoch
        clf('reset')
    end
end

%% Plot final results at optimal epoch
% Test predictions
figure(2)
pl.plPrediction (t, y, ttest, ytestPd, SytestPd, epoch_optim,'r','k')
xlim([t(1) t(end)])
xline(testIdx(1),'--')
title(['Test prediction. Optimal epoch: #' num2str(epoch_optim)]) 

% Hidden states
figure(3)
subplot(5,1,1)
pl.plPrediction ([], [], t, x(1,:)', Sx(1,:)', [],'r','k')
ylabel('x^L')
xlim([t(1) t(end)])
xline(testIdx(1),'--')
title ('Level')
subplot(5,1,2)
pl.plPrediction ([], [], t, x(3,:)', Sx(17,:)', [],'r','k')
ylabel('x^E')
xlim([t(1) t(end)])
xline(testIdx(1),'--')
title ('Exp smoothing')
subplot(5,1,3)
pl.plPrediction (t, [ytrain;yval_nomask;ytest_nomask], t, x(1,:)'+ x(3,:)', Sx(1,:)'+Sx(17,:)'+2*Sx(15,:)', [],'r','k')
ylabel('x^L + x^E')
xlim([t(1) t(end)])
xline(testIdx(1),'--')
title ('Level and exponential smoothing')
subplot(5,1,4)
pl.plPrediction ([], [], t, x(end,:)', Sx(end,:)', [],'r','k')
ylabel('x^{LSTM}')
xlim([t(1) t(end)])
xline(testIdx(1),'--')
title ('LSTM')
subplot(5,1,5)
plot( t, x(6,:)', LineWidth=0.8, Color='k')
% pl.plPrediction ([], [], t, x(6,:)', Sx(41,:)', [],'r','k')
ylabel('x^V')
xlim([t(1) t(end)])
xline(testIdx(1),'--')
title ('Error hidden states')
sgtitle(['Hidden states. Optimal epoch: #' num2str(epoch_optim)]) 























 

 
 
 
 
 
 
 

