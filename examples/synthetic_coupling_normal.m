%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         Synthetic data to exponential smoothing component
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

y1 = 0;
yn = 2*pi;
st = yn/24;
y  = -round(sin(y1:st:yn),2)';
y(end) = [];
y  = repmat(y,[15,1]); 
y = y + randn(size(y))*0.2;

% trend
trend = [linspace(0,2,length(y))]';

y = y+trend;
% plot(y)

t1 = datenum('01-Jan-2000 00:00:00','dd-mmm-yyyy HH:MM:SS');
tend = t1 + 1/24*(size(y,1)-1);
t = [t1:1/24:tend]';
[~, ~, t_.day_month, t_.hour_day] = datevec(t);
[t_.day_week] = weekday(t);
x = [t_.hour_day, t_.day_week];
[x, ~, ~, ~, ~, ~, ~, ~] = dp.normalize(x, [], x, []);
 
%% 2. Option for model
nbobs    = size(y,1);        % total number of observations         
nbtest   = 1*24;             % number of test point
sql      = 12;  
net.sql  = sql;             % Lookback period
net.xsql = 1;
net.nbCov = size(x,2);  % number of covariates
nbtrain  = nbobs-nbtest;     % number of training point
nbval    = 24;

%% 3. Data split and normalizatio
[mytrain, sytrain, ytrain, yval, yval_nomask,...
ytest, ytest_nomask, xtrain, xval, xtest, trainIdx, valIdx, ...
trainValIdx, testIdx] = rnn.RnnDataProcess (x, y, nbobs, nbval, nbtest);

%% 4. State-space models (SSM)
bdlm.comp = [12,7]; % 12: Local trend; 7:LSTM
sQ    = [0,0];      % to construct transition noise matrix Q

% Observation noise decay
sv_up = 1; 
sv_low = 0.2;
nbEpoch_decay = 3;
maxEpoch = 20;
[svGrid] = rnn.svgrid (sv_up, sv_low, nbEpoch_decay, maxEpoch); % create decayed sv over "nbEpoch_decay" epoch
 
% Hidden states for bdlm.comp = [12,7]
seasonality  = 24; % one cyle for data -> hourly: 24

init_x=[
    % #1                                #2          #3   
    % LL                                LT          LSTM 
	  nanmean(ytrain(1:seasonality));	1E-3;	 	0 
];

init_Sx=diag([
    % #1                                #2          #3   
    % LL                                LT          LSTM 
	  1E-3;	                            1E-3;	 	0 
]);

% Build bdlm: contain BDLM matrices, Q, V, hidden states
bdlm  = BDLM.build(bdlm, sQ, sv_up, init_x, init_Sx);

%% 5. Network
% 5.1 Initialize networks
net.learnNoise = 0;
batchSize = 1;
net.sv = [];
% 7: LSTM; just define the number of LSTM layers and #LSTM node
net = rnn.defineNet(net,  sv_up,   batchSize,    maxEpoch,    [7],    [50]);
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
    bdlmT = BDLM.build(bdlm, sQ, svGrid(epoch), xBu_train(:,end), SxBu_train(:,end)); %  build BDLM to again to have correct initial hidden states for valiation
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
    bdlmT = BDLM.build(bdlm, sQ, svGrid(epoch), xBu_val(:,end), SxBu_val(:,end)); %  build BDLM to again to have correct initial hidden states for test set
    [ytestPd_, SytestPd_,~,~, xBp_test, SxBp_test] = task.runHydrid_AGVI(netT, lstmT, bdlmT, xtest, ytest);
    [ytestPd_, SytestPd_]  = dp.denormalize(ytestPd_, SytestPd_, mytrain, sytrain);
    
    % Smoother
    [xBu_train, SxBu_train] = BDLM.KFSmoother_ESM_BNI(bdlm.comp, xBp_train, SxBp_train, xBu_train, SxBu_train, bdlm.A, Czz);
    init_x   = xBu_train(:,1);
    init_Sx  = reshape(SxBu_train(:,1),size(init_x,1),[]);
    init_Sx  = diag(diag(init_Sx));
    bdlm     = BDLM.build(bdlm, sQ, svGrid(epoch), init_x, init_Sx); %  build BDLM to again for the training set of next epoch

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
    pl.plPrediction (t, [ytrain;yval_nomask;ytest_nomask], t, x(1,:)', Sx(1,:)', [],'r','k')
    ylabel('x^L')
    xlim([t(1) valIdx(end)])
    xline(trainIdx(end),'--')
    title ('Level')
    subplot(4,1,3)
    pl.plPrediction ([], [], t, x(2,:)', Sx(5,:)', [],'r','k')
    ylabel('x^T')
    xlim([t(1) valIdx(end)])
    xline(trainIdx(end),'--')
    title ('Trend')
    subplot(4,1,4)
    pl.plPrediction ([], [], t, x(end,:)', Sx(end,:)', [],'r','k')
    ylabel('x^{LSTM}')
    xlim([t(1) valIdx(end)])
    xline(trainIdx(end),'--')
    title ('LSTM')

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
subplot(3,1,1)
pl.plPrediction ([], [], t, x(1,:)', Sx(1,:)', [],'r','k')
ylabel('x^L')
xlim([t(1) t(end)])
xline(testIdx(1),'--')
title ('Level')
subplot(3,1,2)
pl.plPrediction ([], [], t, x(2,:)', Sx(5,:)', [],'r','k')
ylabel('x^T')
xlim([t(1) t(end)])
xline(testIdx(1),'--')
title ('Trend')
subplot(3,1,3)
pl.plPrediction ([], [], t, x(end,:)', Sx(end,:)', [],'r','k')
ylabel('x^{LSTM}')
xlim([t(1) t(end)])
xline(testIdx(1),'--')
title ('LSTM')
sgtitle(['Hidden states. Optimal epoch: #' num2str(epoch_optim)]) 























 

 
 
 
 
 
 
 

