%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         Synthetic data to verify TAGI-LSTM
% Description:  Verify TAGI-LSTM
% Authors:      Van-Dai Vuong, Luong-Ha Nguyen & James-A. Goulet
% Contact:      vuongdai@gmail.com, luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2023 Van-Dai Vuong, Luong-Ha Nguyen & James-A. Goulet 
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
nb_cycles = 10;
y  = repmat(y,[nb_cycles,1]); 

% y = linspace(1,2,length(y))'.*y;
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
sql = 24;  
net.sql  = sql;             % Lookback period
net.xsql = 1;
net.nbCov = size(x,2);  % number of covariates
nbtrain  = nbobs-nbtest;     % number of training point
nbval    = 24;
net.LSTMsmoothing = 1;

%% 3. Data split and normalizatio
[mytrain, sytrain, ytrain, yval, yval_nomask,...
ytest, ytest_nomask, xtrain, xval, xtest, trainIdx, valIdx, ...
trainValIdx, testIdx] = rnn.RnnDataProcess (x, y, nbobs, nbval, nbtest);

% create nan at the beginning of time series; then infer them using LSTM
% smoothing
nb_past_infer = sql*2; % number of points before training to be inferred
net.nb_past_infer = nb_past_infer;
ytrain_raw = ytrain;
nb_nan = nb_past_infer;
ytrain(1:nb_nan) = nan;

%% 4. Network
% 4.1 Observation noise decay
sv_up = 1; 
sv_low = 0.1;
nbEpoch_decay = 5;
maxEpoch = 50;
[svGrid] = rnn.svgrid (sv_up, sv_low, nbEpoch_decay, maxEpoch); % create decayed sv over "nbEpoch_decay" epoch
batchSize = 1;
nb_layer = 1;
net = rnn.defineNet(net,  sv_up,   batchSize,    maxEpoch,    7*ones(1,nb_layer),    40.*ones(1,nb_layer));
netT = net;
net.trainMode = 1;
netT.trainMode = 0;
netT.batchSize = 1;
[net, states, maxIdx, netInfo] = network.initialization(net);

% 4.2 Initialize parameters
theta_save = cell(1,net.maxEpoch+1);
theta_save{1} = tagi.initializeWeightBias(net);

% 4.3 Initialize memory for LSTM: cell and hidden states
m_Mem = 1;    % initialized values for mh and mc (cell and hidden states)
S_Mem = 1;    % initialized values for Sh and Sc (cell and hidden states)
% Initialize LSTM's memory at t=0, epoch=1
% Mem{1} = mh (means for hidden states)
% Mem{2} = Sh (variances for hidden states)
% Mem{3} = mc (means for cell states)
% Mem{4} = Sc (variances for cell states)
Mem0_infer = rnn.initializeRnnMemory_v1 (net.layer, net.nodes, net.batchSize, m_Mem, S_Mem);

% 4.4 Initialize initial sequence length
m_Sq = ones(sql,1); % mean
S_Sq = ones(sql,1); % variances
Sq_infer{1} = m_Sq; 
Sq_infer{2} = S_Sq;

%% 5. Analyze
disp(['Training................'])
for epoch = 1:maxEpoch
    % load observation noise
    net.sv   = svGrid(epoch);
    netT.sv  = svGrid(epoch);
    % Sequence length
    Sq{1} = Sq_infer{1}(1:sql);
    Sq{2} = Sq_infer{2}(1:sql);

    % Train
    [xtrain_loop, ytrain_loop] = tagi.prepDataBatch_RNN (xtrain, ytrain, batchSize, sql);
    lstm = [];
    lstm.Sq = Sq;
    lstm.Mem = Mem0_infer;
    lstm.theta = theta_save{epoch};
    [ytrainPd, SytrainPd, theta, memVal,~,~, Mem0_infer, Sq_infer] = task.runLSTM(net, lstm, xtrain_loop, ytrain_loop);

    % Validation
    lstmT = [];
    lstmT.Mem = memVal;
    lstmT.theta = theta;
    lstmT.Sq  = rnn.getSq (sql, ytrain, zeros(size(ytrain)));
    [yvalPd_, SyvalPd_, ~, memTest] = task.runLSTM(netT, lstmT, xval, yval);

    figure (2)
    y_norm = [ytrain; yval_nomask];
    ttt  = [1:1:nb_past_infer];
    pl.plPrediction (trainValIdx', y_norm, valIdx', yvalPd_, SyvalPd_, epoch,'r','k')
    pl.plPrediction (ttt, ytrain_raw(1:nb_past_infer,:), ttt, Sq_infer{1}, Sq_infer{2}, epoch, 'r','b')
    xline(nb_past_infer+1,'--')
    xline(valIdx(1),'--')
    xlim ([1 trainValIdx(end)])
    legend('observation','validation','','','inferred history','Location', 'north')
    pause(0.01)
    if epoch<maxEpoch
        clf('reset')
    end

    % Evaluate validation
    [yvalPd_, SyvalPd_] = dp.denormalize(yvalPd_, SyvalPd_, mytrain, sytrain);
    LL = gather(mt.loglik(y(valIdx), yvalPd_, SyvalPd_));
    disp(['Epoch ' num2str(epoch) '/' num2str(maxEpoch) '. LL :' num2str(LL)])

    % save for records
    theta_save{epoch+1} = theta;
    LL_save(epoch)      = LL;
end





















 

 
 
 
 
 
 
 

