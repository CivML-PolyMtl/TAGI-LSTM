% Calculate metrics for the traffic dataset
clc
close all
clear all
path  = char([cd]);

seasonality = 24;
%% Train/test split 2008_01_14
% load data
data   = load(char([path, '/data/traffic.mat']));
y     = data.traffic(137:472,:);
ytrain = y(1:end-168,:);     % observation
ytest  = y(end-168+1:end,:); % observation

% load result
tagi = load(char([path, '/result/traffic/traffic_2008_01_14.mat']));
ytestPd  = tagi.ytestPd;
SytestPd = tagi.SytestPd;
p50_tagi = mt.computeND(ytest,ytestPd);
p90_tagi = mt.compute90QL (ytest, ytestPd, SytestPd);
RMSE_tagi = mt.computeRMSE(ytest,ytestPd);
MASE_tagi = mt.computeMASE(ytest, ytestPd, ytrain,seasonality);

disp('traffic train/test split 2008_01_14 ...........')
disp(['ND/p50:    ' num2str(p50_tagi)])
disp(['p90:    ' num2str(p90_tagi)])
disp(['RMSE:    ' num2str(RMSE_tagi)])
disp(['MASE:    ' num2str(MASE_tagi)])
disp('............................................')


%% Train/test split 2008_06_15
data   = load(char([path, '/data/traffic.mat']));
y     = data.traffic(1:4169,:);
ytrain = y(1:end-168,:);      % observation
ytest  = y(end-168+1:end,:);  % observation

% load result
tagi = load(char([path, '/result/traffic/traffic_2008_06_15.mat']));
ytestPd  = tagi.ytestPd;
SytestPd = tagi.SytestPd;
p50_tagi = mt.computeND(ytest,ytestPd);
p90_tagi = mt.compute90QL (ytest, ytestPd, SytestPd);
RMSE_tagi = mt.computeRMSE(ytest,ytestPd);
MASE_tagi = mt.computeMASE(ytest, ytestPd, ytrain,seasonality);

disp('traffic train/test split 2008_06_15 ...........')
disp(['ND/p50:    ' num2str(p50_tagi)])
disp(['p90:    ' num2str(p90_tagi)])
disp(['RMSE:    ' num2str(RMSE_tagi)])
disp(['MASE:    ' num2str(MASE_tagi)])
disp('............................................')

%% Train/test split last_7_days
data   = load(char([path, '/data/traffic.mat']));
y     = data.traffic;
ytrain = y(1:end-168,:);      % observation
ytest  = y(end-168+1:end,:);  % observation

tagi = load(char([path, '/result/traffic/traffic_last_7_days.mat']));
ytestPd  = tagi.ytestPd;
SytestPd = tagi.SytestPd;
p50_tagi = mt.computeND(ytest,ytestPd);
p90_tagi = mt.compute90QL (ytest, ytestPd, SytestPd);
RMSE_tagi = mt.computeRMSE(ytest,ytestPd);
MASE_tagi = mt.computeMASE(ytest, ytestPd, ytrain,seasonality);

disp('traffic train/test split last 7 days ...........')
disp(['ND/p50:    ' num2str(p50_tagi)])
disp(['p90:    ' num2str(p90_tagi)])
disp(['RMSE:    ' num2str(RMSE_tagi)])
disp(['MASE:    ' num2str(MASE_tagi)])















