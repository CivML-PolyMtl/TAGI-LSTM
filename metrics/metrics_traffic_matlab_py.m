% Calculate metrics for the electricity dataset
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
disp('matlab.............')
disp(['ND/p50:    ' num2str(p50_tagi)])
disp(['p90:    ' num2str(p90_tagi)])
disp(['RMSE:    ' num2str(RMSE_tagi)])
disp(['MASE:    ' num2str(MASE_tagi)])
disp('............................................')

% load result
ytestPd = readtable(char([path, '/result/traffic/traffic_2008_01_14_ytestPd_pyTAGI.csv']));
ytestPd = table2array(ytestPd);
SytestPd = readtable(char([path, '/result/traffic/traffic_2008_01_14_SytestPd_pyTAGI.csv']));
SytestPd = table2array(SytestPd);


p50_tagi = mt.computeND(ytest,ytestPd);
p90_tagi = mt.compute90QL (ytest, ytestPd, SytestPd);
RMSE_tagi = mt.computeRMSE(ytest,ytestPd);
MASE_tagi = mt.computeMASE(ytest, ytestPd, ytrain,seasonality);

disp('pytagi...........')
disp(['ND/p50:    ' num2str(p50_tagi)])
disp(['p90:    ' num2str(p90_tagi)])
disp(['RMSE:    ' num2str(RMSE_tagi)])
disp(['MASE:    ' num2str(MASE_tagi)])
disp('............................................')

% %% Train/test split 2014_09_01
% data   = load(char([path, '/data/electricity.mat']));
% y      = data.elec(17545:23544,:);
% ytrain = y(1:end-168,:);      % observation
% ytest  = y(end-168+1:end,:);  % observation
% 
% % load result
% tagi = load(char([path, '/result/electricity/electricity_2014_09_01.mat']));
% ytestPd  = tagi.ytestPd;
% SytestPd = tagi.SytestPd;
% p50_tagi = mt.computeND(ytest,ytestPd);
% p90_tagi = mt.compute90QL (ytest, ytestPd, SytestPd);
% RMSE_tagi = mt.computeRMSE(ytest,ytestPd);
% MASE_tagi = mt.computeMASE(ytest, ytestPd, ytrain,seasonality);
% 
% disp('electricity train/test split 2014_09_01 ...........')
% disp(['ND/p50:    ' num2str(p50_tagi)])
% disp(['p90:    ' num2str(p90_tagi)])
% disp(['RMSE:    ' num2str(RMSE_tagi)])
% disp(['MASE:    ' num2str(MASE_tagi)])
% disp('............................................')
% 
% %% Train/test split last_7_days
% data   = load(char([path, '/data/electricity.mat']));
% y      = data.elec;
% ytrain = y(1:end-168,:);     % observation
% ytest  = y(end-168+1:end,:); % observation
% 
% tagi = load(char([path, '/result/electricity/electricity_last_7_days.mat']));
% ytestPd  = tagi.ytestPd;
% SytestPd = tagi.SytestPd;
% p50_tagi = mt.computeND(ytest,ytestPd);
% p90_tagi = mt.compute90QL (ytest, ytestPd, SytestPd);
% RMSE_tagi = mt.computeRMSE(ytest,ytestPd);
% MASE_tagi = mt.computeMASE(ytest, ytestPd, ytrain,seasonality);
% 
% disp('electricity train/test split last 7 days ...........')
% disp(['ND/p50:    ' num2str(p50_tagi)])
% disp(['p90:    ' num2str(p90_tagi)])
% disp(['RMSE:    ' num2str(RMSE_tagi)])
% disp(['MASE:    ' num2str(MASE_tagi)])















